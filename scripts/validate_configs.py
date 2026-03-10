#!/usr/bin/env python3
"""Validate all Hydra config combinations.

Loads every config group and their combinations using hydra.initialize()
and asserts no missing interpolations or composition errors.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Resolve the absolute config dir relative to this script
CONFIGS_DIR = str((Path(__file__).resolve().parent.parent / "configs").resolve())

# Config groups and their options
CONFIG_GROUPS: dict[str, list[str]] = {
    "sim": ["numpy_quad", "hover"],
    "control": ["ppo", "ppo_asymmetric"],
    "perception": ["none", "noise_injection"],
    "reward": ["monorace", "perception_aware"],
    "domain_rand": ["none", "selective", "uniform_30pct"],
    "logging": ["wandb", "tensorboard"],
}

# Experiment overrides (loaded on top of train.yaml)
EXPERIMENTS: list[str] = [
    "monorace_baseline",
    "perception_noise",
    "asymmetric_critic",
    "ablation_no_dr",
]

# Root configs
ROOT_CONFIGS: list[str] = ["train", "eval"]


def resolve_cfg(cfg: DictConfig) -> None:
    """Resolve all interpolations in the config, raising on missing values.

    Skips keys that are marked as mandatory (???) since those are expected
    to be provided at runtime.
    """
    for key in cfg:
        try:
            val = cfg[key]
            if isinstance(val, DictConfig):
                resolve_cfg(val)
        except Exception as exc:
            # MissingMandatoryValue is expected for ??? fields
            if "MissingMandatoryValue" in type(exc).__name__:
                continue
            raise


def validate_single_group(group: str, option: str) -> bool:
    """Validate a single config group option loads correctly."""
    try:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=[f"{group}={option}"])
            resolve_cfg(cfg)
        return True
    except Exception as exc:
        print(f"  FAIL: {group}={option}: {exc}")
        return False


def validate_root_config(config_name: str) -> bool:
    """Validate a root config loads correctly."""
    try:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            # eval.yaml has checkpoint_path=??? which is a mandatory override;
            # skip it for validation
            overrides = []
            if config_name == "eval":
                overrides.append("checkpoint_path=/tmp/dummy.pt")
            cfg = compose(config_name=config_name, overrides=overrides)
            resolve_cfg(cfg)
        return True
    except Exception as exc:
        print(f"  FAIL: root config '{config_name}': {exc}")
        return False


def validate_experiment(experiment: str) -> bool:
    """Validate an experiment override composes correctly on top of train.yaml."""
    try:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[f"+experiment={experiment}"],
            )
            resolve_cfg(cfg)
        return True
    except Exception as exc:
        print(f"  FAIL: experiment/{experiment}: {exc}")
        return False


def validate_all_combinations() -> bool:
    """Validate a representative set of cross-group combinations."""
    passed = True

    # Test all pairs of (sim, control) and (domain_rand, perception)
    # to keep combinatorial explosion manageable
    for sim_opt, ctrl_opt in itertools.product(
        CONFIG_GROUPS["sim"], CONFIG_GROUPS["control"]
    ):
        for dr_opt, perc_opt in itertools.product(
            CONFIG_GROUPS["domain_rand"], CONFIG_GROUPS["perception"]
        ):
            overrides = [
                f"sim={sim_opt}",
                f"control={ctrl_opt}",
                f"perception={perc_opt}",
                f"domain_rand={dr_opt}",
            ]
            label = ", ".join(overrides)
            try:
                with initialize_config_dir(
                    config_dir=CONFIGS_DIR, version_base=None
                ):
                    cfg = compose(config_name="train", overrides=overrides)
                    resolve_cfg(cfg)
            except Exception as exc:
                print(f"  FAIL: [{label}]: {exc}")
                passed = False

    return passed


def main() -> int:
    total_pass = 0
    total_fail = 0

    # 1. Validate root configs
    print("=== Root configs ===")
    for config_name in ROOT_CONFIGS:
        ok = validate_root_config(config_name)
        if ok:
            print(f"  OK: {config_name}.yaml")
            total_pass += 1
        else:
            total_fail += 1

    # 2. Validate each group option individually
    print("\n=== Config groups ===")
    for group, options in CONFIG_GROUPS.items():
        for option in options:
            ok = validate_single_group(group, option)
            if ok:
                print(f"  OK: {group}/{option}.yaml")
                total_pass += 1
            else:
                total_fail += 1

    # 3. Validate experiments
    print("\n=== Experiments ===")
    for experiment in EXPERIMENTS:
        ok = validate_experiment(experiment)
        if ok:
            print(f"  OK: experiment/{experiment}.yaml")
            total_pass += 1
        else:
            total_fail += 1

    # 4. Validate cross-group combinations
    print("\n=== Cross-group combinations ===")
    if validate_all_combinations():
        n_combos = (
            len(CONFIG_GROUPS["sim"])
            * len(CONFIG_GROUPS["control"])
            * len(CONFIG_GROUPS["domain_rand"])
            * len(CONFIG_GROUPS["perception"])
        )
        print(f"  OK: all {n_combos} combinations passed")
        total_pass += n_combos
    else:
        total_fail += 1

    # Summary
    print(f"\n{'='*40}")
    print(f"Results: {total_pass} passed, {total_fail} failed")

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
