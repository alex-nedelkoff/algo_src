"""TrainingLoop protocol for pluggable training loop implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from omegaconf import DictConfig

if TYPE_CHECKING:
    from artifacts.uploader import ArtifactUploader


class TrainingLoop(Protocol):
    """Protocol that all training loops must satisfy.

    Implementations are instantiated by Hydra via ``cfg.loop._target_``
    and receive the full config + an optional artifact uploader at
    ``run()`` time.
    """

    def run(self, cfg: DictConfig, uploader: ArtifactUploader | None = None) -> None: ...
