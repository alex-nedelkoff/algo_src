"""Equivalence check: torch_quad must match numpy_quad to float precision.

Since the PyTorch port is intended to be a drop-in for system identification,
both implementations must produce identical state trajectories given the same
initial state, action sequence, and parameters. Float64 throughout, so the
tolerance can be tight.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim.dynamics.numpy_quad import NumpyQuadDynamics
from sim.dynamics.params import VehicleParams
from sim.dynamics.torch_params import TorchVehicleParams
from sim.dynamics.torch_quad import step as torch_step


_NUMPY_NEXT_RTOL = 1e-12
_NUMPY_NEXT_ATOL = 1e-12
_ROLLOUT_RTOL = 1e-9
_ROLLOUT_ATOL = 1e-9


def _hover_state(p: VehicleParams) -> np.ndarray:
    """Standard hover initial state matching numpy_quad's reset() output."""
    s = np.zeros(17, dtype=np.float64)
    s[2] = 1.0      # z = 1m
    s[6] = 1.0      # quat w
    hover_omega = float(np.sqrt(p.mass * 9.81 / (4.0 * p.k_thrust)))
    s[13:17] = hover_omega
    return s


def _np_step_one(dyn: NumpyQuadDynamics, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
    """Wrap numpy_quad's batched step() to a single-env interface."""
    return dyn.step(state[None, :], action[None, :], dt=dt)[0]


def test_step_at_hover_matches():
    p = VehicleParams()
    np_dyn = NumpyQuadDynamics(p, dt=0.01)
    np_dyn.reset(1)

    state = _hover_state(p)
    action = np.full(4, np_dyn.hover_omega(), dtype=np.float64)

    np_next = _np_step_one(np_dyn, state, action, dt=0.01)

    torch_p = TorchVehicleParams.from_vehicle_params(p)
    torch_next = torch_step(
        torch.from_numpy(state), torch.from_numpy(action), torch_p, dt=0.01,
    ).detach().numpy()

    np.testing.assert_allclose(torch_next, np_next, rtol=_NUMPY_NEXT_RTOL, atol=_NUMPY_NEXT_ATOL)


def test_step_random_state_action_matches():
    rng = np.random.default_rng(42)
    p = VehicleParams()
    np_dyn = NumpyQuadDynamics(p, dt=0.005)
    np_dyn.reset(1)

    # Random state with a normalised quaternion + nonzero everything else
    state = np.zeros(17)
    state[0:3] = rng.uniform(-2.0, 2.0, size=3)
    state[3:6] = rng.uniform(-1.0, 1.0, size=3)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    state[6:10] = q
    state[10:13] = rng.uniform(-2.0, 2.0, size=3)
    state[13:17] = rng.uniform(0.0, np_dyn.hover_omega() * 1.5, size=4)

    action = rng.uniform(0.0, p.max_omega, size=4)

    np_next = _np_step_one(np_dyn, state, action, dt=0.005)

    torch_p = TorchVehicleParams.from_vehicle_params(p)
    torch_next = torch_step(
        torch.from_numpy(state), torch.from_numpy(action), torch_p, dt=0.005,
    ).detach().numpy()

    np.testing.assert_allclose(torch_next, np_next, rtol=_NUMPY_NEXT_RTOL, atol=_NUMPY_NEXT_ATOL)


def test_long_rollout_matches():
    """100-step rollout of identical (state, action, params) — drift must stay tiny."""
    rng = np.random.default_rng(7)
    p = VehicleParams()
    dt = 0.01
    np_dyn = NumpyQuadDynamics(p, dt=dt)
    np_dyn.reset(1)

    state_np = _hover_state(p)
    state_t = torch.from_numpy(state_np.copy())
    torch_p = TorchVehicleParams.from_vehicle_params(p)

    hover = np_dyn.hover_omega()
    for _ in range(100):
        # Random small perturbation around hover so it actually moves
        action = hover + rng.uniform(-100.0, 100.0, size=4)
        action = np.clip(action, 0.0, p.max_omega)

        state_np = _np_step_one(np_dyn, state_np, action, dt=dt)
        state_t = torch_step(state_t, torch.from_numpy(action), torch_p, dt=dt)

    np.testing.assert_allclose(
        state_t.detach().numpy(), state_np, rtol=_ROLLOUT_RTOL, atol=_ROLLOUT_ATOL,
    )


def test_grad_flows_through_params():
    """All 8 learnable scalars must receive non-trivial gradients on a rollout loss.

    Action and initial omega are deliberately asymmetric so each component of
    the inertia tensor sees nonzero torque on this single step.
    """
    p = VehicleParams()
    torch_p = TorchVehicleParams.from_vehicle_params(p)

    hover = float(np.sqrt(p.mass * 9.81 / (4.0 * p.k_thrust)))

    # Nonzero body rates so the gyroscopic term excites the inertia params.
    state_np = _hover_state(p).copy()
    state_np[10:13] = [0.5, -0.3, 0.8]
    state = torch.from_numpy(state_np)

    # Asymmetric motor commands so roll/pitch/yaw torques are all nonzero.
    action = torch.tensor(
        [hover + 80.0, hover - 60.0, hover + 40.0, hover - 100.0], dtype=torch.float64
    )

    # Two-step rollout so motor lag (tau_motor) fully participates in the loss.
    next_state = torch_step(state, action, torch_p, dt=0.01)
    next_state = torch_step(next_state, action, torch_p, dt=0.01)
    loss = next_state.pow(2).sum()
    loss.backward()

    for name, prm in torch_p.named_parameters():
        assert prm.grad is not None, f"{name}: no grad"
        assert torch.isfinite(prm.grad).all(), f"{name}: non-finite grad"
        assert prm.grad.abs().item() > 0.0, f"{name}: zero grad"
