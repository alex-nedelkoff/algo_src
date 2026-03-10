"""GateNet — U-Net architecture stub for gate segmentation and corner regression.

This module requires ``torch``.  The import is gated so that the rest of the
perception package remains usable without PyTorch installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = ["GateNet", "TORCH_AVAILABLE"]

TORCH_AVAILABLE = _TORCH_AVAILABLE


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "GateNet requires PyTorch.  Install with:  pip install torch torchvision"
        )


if _TORCH_AVAILABLE:

    class _EncoderBlock(nn.Module):
        """Two 3x3 conv layers + batch norm + ReLU."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class _DecoderBlock(nn.Module):
        """Upconvolution + concatenation with skip connection + double conv."""

        def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
            super().__init__()
            self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.block = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.upconv(x)
            # Handle spatial size mismatch from non-power-of-2 inputs
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    class GateNet(nn.Module):
        """U-Net stub for gate segmentation and corner coordinate regression.

        Input:
            (batch, 3, H, W) image tensor.

        Output:
            Tuple of:
                - segmentation_mask: (batch, 1, H, W)
                - corner_coords: (batch, 4, 2) — four 2D corner positions
        """

        def __init__(self, base_channels: int = 32) -> None:
            super().__init__()
            c = base_channels

            # Encoder
            self.enc1 = _EncoderBlock(3, c)
            self.enc2 = _EncoderBlock(c, c * 2)
            self.enc3 = _EncoderBlock(c * 2, c * 4)
            self.enc4 = _EncoderBlock(c * 4, c * 8)
            self.pool = nn.MaxPool2d(2)

            # Bottleneck
            self.bottleneck = _EncoderBlock(c * 8, c * 16)

            # Decoder
            self.dec4 = _DecoderBlock(c * 16, c * 8, c * 8)
            self.dec3 = _DecoderBlock(c * 8, c * 4, c * 4)
            self.dec2 = _DecoderBlock(c * 4, c * 2, c * 2)
            self.dec1 = _DecoderBlock(c * 2, c, c)

            # Segmentation head
            self.seg_head = nn.Conv2d(c, 1, 1)

            # Corner regression head — global average pool then FC
            self.corner_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 8),  # 4 corners x 2 coords
            )

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Encoder path
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            # Bottleneck
            b = self.bottleneck(self.pool(e4))

            # Decoder path
            d4 = self.dec4(b, e4)
            d3 = self.dec3(d4, e3)
            d2 = self.dec2(d3, e2)
            d1 = self.dec1(d2, e1)

            # Heads
            seg_mask = torch.sigmoid(self.seg_head(d1))
            corners = self.corner_head(d1).view(-1, 4, 2)

            return seg_mask, corners

else:
    # Placeholder when torch is not installed
    class GateNet:  # type: ignore[no-redef]
        """Stub placeholder — PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            _check_torch()
