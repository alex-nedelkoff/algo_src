"""Tests for the DAgger student evaluation script."""
from __future__ import annotations

from pathlib import Path

import torch

from scripts.evaluate_student import build_student


class TestBuildStudent:
    """Test student network loading from .pt files."""

    def test_load_student_from_pt(self, tmp_path: Path) -> None:
        """Round-trip: save and load a student network."""
        model = torch.nn.Sequential(
            torch.nn.Linear(24, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
            torch.nn.Sigmoid(),
        )
        pt_path = tmp_path / "student_test.pt"
        torch.save(model.state_dict(), pt_path)

        loaded = build_student(str(pt_path))
        obs = torch.randn(1, 24)
        with torch.no_grad():
            action = loaded(obs)
        assert action.shape == (1, 4)
        # Sigmoid output -> [0, 1]
        assert (action >= 0).all() and (action <= 1).all()

    def test_output_deterministic(self, tmp_path: Path) -> None:
        """Student in eval mode produces deterministic output."""
        model = torch.nn.Sequential(
            torch.nn.Linear(24, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
            torch.nn.Sigmoid(),
        )
        pt_path = tmp_path / "student_test.pt"
        torch.save(model.state_dict(), pt_path)

        loaded = build_student(str(pt_path))
        obs = torch.randn(1, 24)
        with torch.no_grad():
            a1 = loaded(obs)
            a2 = loaded(obs)
        assert torch.allclose(a1, a2)
