"""ONNX export utility for trained models.

This module provides utilities for exporting trained PyTorch models
to ONNX format for deployment on the Jetson Orin NX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_to_onnx(
    model: Any,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 17),
    opset_version: int = 17,
) -> Path:
    """Export a trained model to ONNX format.

    Args:
        model: Trained PyTorch model to export.
        output_path: Path to save the .onnx file.
        input_shape: Shape of the model input tensor.
        opset_version: ONNX opset version to use.

    Returns:
        Path to the exported ONNX file.

    Raises:
        NotImplementedError: This is a stub; full implementation pending.
    """
    raise NotImplementedError(
        "ONNX export not yet implemented. "
        "Requires torch and onnx dependencies. "
        "See COR project board for the export task."
    )
