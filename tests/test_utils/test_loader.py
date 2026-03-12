"""Tests for utils.tb_check.loader."""

from __future__ import annotations

import os
import tempfile

import pytest

from utils.tb_check.loader import load_run


def _write_events(log_dir: str, scalars: dict[str, list[tuple[int, float]]]) -> None:
    """Write a synthetic TensorBoard event file using SummaryWriter.

    Args:
        log_dir: Directory to write the event file to.
        scalars: Mapping of tag to list of (step, value) tuples.
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=log_dir)
    for tag, step_values in scalars.items():
        for step, value in step_values:
            writer.add_scalar(tag, value, global_step=step)
    writer.close()


class TestLoadRunColumns:
    """Verify the output DataFrame schema."""

    def test_columns_present(self, tmp_path: pytest.TempPathFactory) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0), (1, 2.0)]})
        df = load_run(str(tmp_path))
        assert set(df.columns) == {"step", "tag", "value"}

    def test_returns_correct_values(self, tmp_path: pytest.TempPathFactory) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.5), (10, 3.0)]})
        df = load_run(str(tmp_path))
        subset = df[df["tag"] == "rollout/ep_rew_mean"].sort_values("step")
        assert len(subset) == 2
        assert float(subset["value"].iloc[0]) == pytest.approx(1.5, abs=1e-4)
        assert float(subset["value"].iloc[1]) == pytest.approx(3.0, abs=1e-4)

    def test_multiple_tags(self, tmp_path: pytest.TempPathFactory) -> None:
        _write_events(
            str(tmp_path),
            {
                "rollout/ep_rew_mean": [(0, 1.0)],
                "train/approx_kl": [(0, 0.01)],
            },
        )
        df = load_run(str(tmp_path))
        assert set(df["tag"].unique()) == {"rollout/ep_rew_mean", "train/approx_kl"}


class TestLoadRunLastN:
    """Verify last_n step-filtering behaviour."""

    def test_last_n_filters_early_steps(self, tmp_path: pytest.TempPathFactory) -> None:
        steps = list(range(0, 1000, 100))
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(s, float(s)) for s in steps]})
        df = load_run(str(tmp_path), last_n=300)
        # max_step = 900; keep steps >= 900 - 300 = 600
        assert df["step"].min() >= 600

    def test_last_n_none_returns_all_steps(self, tmp_path: pytest.TempPathFactory) -> None:
        steps = list(range(0, 500, 50))
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(s, 1.0) for s in steps]})
        df = load_run(str(tmp_path), last_n=None)
        assert len(df) == len(steps)


class TestLoadRunEmpty:
    """Verify graceful handling of empty or missing directories."""

    def test_nonexistent_directory_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does_not_exist")
            df = load_run(missing)
        assert list(df.columns) == ["step", "tag", "value"]
        assert len(df) == 0

    def test_empty_directory_returns_empty(self, tmp_path: pytest.TempPathFactory) -> None:
        df = load_run(str(tmp_path))
        assert list(df.columns) == ["step", "tag", "value"]
        assert df.empty
