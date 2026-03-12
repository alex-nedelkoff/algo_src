"""Training entrypoint for drone racing.

Single Hydra entrypoint that handles cross-cutting concerns (config,
seeding, W&B init, artifact uploading) and delegates the actual
training to a pluggable :class:`TrainingLoop` implementation selected
via ``cfg.loop._target_``.

Usage:
    python -m training                              # default config
    python -m training +experiment=monorace_baseline # experiment override
    python -m training sim.n_envs=50                # CLI override
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env from project root (best-effort, no dependency)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    import os

    with env_path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Training entrypoint."""
    _load_dotenv()
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)

    # W&B init (optional)
    uploader = None
    wandb_initialized = False
    if cfg.logging.get("backend") == "wandb":
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.logging.project,
                entity=cfg.logging.get("entity"),
                tags=list(cfg.logging.get("tags", [])),
                group=cfg.logging.get("group"),
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
            )
            wandb_initialized = True

            from artifacts.uploader import ArtifactUploader

            uploader = ArtifactUploader(run_id=wandb_run.id)
        except ImportError as exc:
            log.warning("Import failed, skipping W&B logging: %s", exc)

    # Instantiate and run the training loop
    loop = hydra.utils.instantiate(cfg.loop)
    loop.run(cfg, uploader=uploader)

    # Cleanup
    if uploader is not None:
        uploader.close(timeout=60)

    if wandb_initialized:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
