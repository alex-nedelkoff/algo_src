"""Tests for autoresearch configuration."""

from autoresearch.config import AutoResearchConfig


def test_default_config():
    cfg = AutoResearchConfig()
    assert cfg.wandb_project == "corvidx-drone-racing"
    assert cfg.mode == "interactive"
    assert cfg.budgets["hyperparameter"] == 5_000_000
    assert cfg.constraints["min_success_rate"] == 0.8
    assert cfg.branch_selection["exploit_weight"] == 1.0
    assert cfg.coordination["claim_timeout_hours"] == 4


def test_config_override():
    cfg = AutoResearchConfig(mode="yolo", budgets={"hyperparameter": 1_000_000})
    assert cfg.mode == "yolo"
    assert cfg.budgets["hyperparameter"] == 1_000_000
