"""Supervised training loop stub.

Placeholder for future supervised perception training (e.g., gate
detection CNN).  Selected via ``loop: supervised`` in Hydra config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from artifacts.uploader import ArtifactUploader


class SupervisedTrainingLoop:
    """Supervised training loop (not yet implemented)."""

    def run(self, cfg: DictConfig, uploader: ArtifactUploader | None = None) -> None:
        raise NotImplementedError("Supervised training loop not yet implemented")
