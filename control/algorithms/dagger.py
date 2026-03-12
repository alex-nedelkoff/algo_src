"""DAgger (Dataset Aggregation) for teacher-student distillation.

The teacher (Phase 3 asymmetric policy) provides action labels.
The student learns to map EKF-filtered 24D obs to 4D actions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class StudentNetwork(nn.Module):
    """Simple MLP matching Phase 1 architecture."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        layers.append(nn.Sigmoid())  # actions in [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DAggerTrainer:
    """DAgger training loop for student policy."""

    def __init__(
        self,
        obs_dim: int = 24,
        act_dim: int = 4,
        hidden: Optional[list[int]] = None,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        if hidden is None:
            hidden = [64, 64, 64]
        self.device = torch.device(device)
        self.student = StudentNetwork(obs_dim, act_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self._obs_buffer: list[np.ndarray] = []
        self._act_buffer: list[np.ndarray] = []

    def add_data(self, obs: np.ndarray, actions: np.ndarray) -> None:
        self._obs_buffer.append(obs)
        self._act_buffer.append(actions)

    def train_epoch(
        self,
        obs: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        batch_size: int = 256,
    ) -> float:
        if obs is None:
            obs = np.concatenate(self._obs_buffer, axis=0)
            actions = np.concatenate(self._act_buffer, axis=0)

        obs_t = torch.from_numpy(obs).float().to(self.device)
        act_t = torch.from_numpy(actions).float().to(self.device)

        dataset = torch.utils.data.TensorDataset(obs_t, act_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.student.train()
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_act in loader:
            pred = self.student(batch_obs)
            loss = self.loss_fn(pred, batch_act)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.student.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(self.device)
            actions = self.student(obs_t)
        return actions.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.student.state_dict(), path)

    def load(self, path: str) -> None:
        self.student.load_state_dict(torch.load(path, weights_only=True))
