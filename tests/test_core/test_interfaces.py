"""Tests for core.interfaces module — ABC enforcement."""

from __future__ import annotations

import pytest

from core.interfaces import Detector, Dynamics, Policy, RewardFn
from core.types import Action, Observation, QuadState


class TestDynamicsABC:
    """Tests that Dynamics cannot be instantiated without implementing all methods."""

    def test_cannot_instantiate_directly(self) -> None:
        """Dynamics ABC should raise TypeError if instantiated directly."""
        with pytest.raises(TypeError):
            Dynamics()  # type: ignore[abstract]

    def test_partial_implementation_raises(self) -> None:
        """Partial implementation should still raise TypeError."""

        class PartialDynamics(Dynamics):
            def reset(self, state: QuadState | None = None) -> QuadState:
                return QuadState()

        with pytest.raises(TypeError):
            PartialDynamics()  # type: ignore[abstract]

    def test_full_implementation_works(self) -> None:
        """Full implementation should be instantiable."""

        class FullDynamics(Dynamics):
            def reset(self, state: QuadState | None = None) -> QuadState:
                return QuadState()

            def step(self, state: QuadState, action: Action, dt: float) -> QuadState:
                return state

        dyn = FullDynamics()
        assert dyn.reset() is not None


class TestPolicyABC:
    """Tests that Policy cannot be instantiated without implementing predict."""

    def test_cannot_instantiate_directly(self) -> None:
        """Policy ABC should raise TypeError if instantiated directly."""
        with pytest.raises(TypeError):
            Policy()  # type: ignore[abstract]

    def test_full_implementation_works(self) -> None:
        """Full implementation should be instantiable."""

        class FullPolicy(Policy):
            def predict(self, observation: Observation) -> Action:
                return Action()

        policy = FullPolicy()
        obs = Observation(state=QuadState())
        assert policy.predict(obs) is not None


class TestDetectorABC:
    """Tests that Detector cannot be instantiated without implementing detect."""

    def test_cannot_instantiate_directly(self) -> None:
        """Detector ABC should raise TypeError if instantiated directly."""
        with pytest.raises(TypeError):
            Detector()  # type: ignore[abstract]


class TestRewardFnABC:
    """Tests that RewardFn cannot be instantiated without implementing compute."""

    def test_cannot_instantiate_directly(self) -> None:
        """RewardFn ABC should raise TypeError if instantiated directly."""
        with pytest.raises(TypeError):
            RewardFn()  # type: ignore[abstract]
