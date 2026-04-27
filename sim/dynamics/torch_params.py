"""Learnable PyTorch quadrotor parameters for system identification.

Mirrors the 8 scalar parameters that drive ``NumpyQuadDynamics.step()``:

    mass        : kg
    Ixx, Iyy, Izz : kg * m^2 (diagonal inertia tensor)
    k_thrust    : N / (rad/s)^2
    k_torque    : N * m / (rad/s)^2
    arm_length  : m
    tau_motor   : s (first-order motor time constant)

Stored internally as ``log(value)`` (one ``nn.Parameter`` each) and exposed
through Python properties as the original positive scalar. Log-space
parameterisation keeps Adam well-behaved across the 7-order-of-magnitude
spread of the physical scales (k_torque ~ 1e-10 vs mass ~ 1e-2): an additive
step of size ``lr`` in log space is a multiplicative factor of ``exp(lr)``
on the underlying scalar, so a learning rate that's reasonable for one
parameter is reasonable for all of them. As a side effect, scalars stay
strictly positive without any clamping.

``drag_coeff`` and ``max_rpm`` from the numpy ``VehicleParams`` are
deliberately held fixed for the grey-box MVP — drag is assumed zero and
``max_rpm`` is a hard clip used only at the dynamics boundary.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from sim.dynamics.params import VehicleParams

_PARAM_NAMES = ("mass", "Ixx", "Iyy", "Izz", "k_thrust", "k_torque", "arm_length", "tau_motor")


class TorchVehicleParams(nn.Module):
    """Learnable container for the 8 scalar parameters of the grey-box model.

    Each scalar is parameterised as its natural log so updates are
    multiplicative on the underlying value.
    """

    def __init__(
        self,
        mass: float,
        Ixx: float,
        Iyy: float,
        Izz: float,
        k_thrust: float,
        k_torque: float,
        arm_length: float,
        tau_motor: float,
        max_omega: float,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        for name, value in zip(
            _PARAM_NAMES,
            (mass, Ixx, Iyy, Izz, k_thrust, k_torque, arm_length, tau_motor),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            self.register_parameter(
                f"_log_{name}", nn.Parameter(torch.tensor(math.log(value), dtype=dtype)),
            )
        # Fixed (not learned): used only as the action clip ceiling.
        self.register_buffer("max_omega", torch.tensor(max_omega, dtype=dtype))

    # --- Live, autograd-friendly views of each scalar (always > 0) -----------
    @property
    def mass(self) -> torch.Tensor: return torch.exp(self._log_mass)
    @property
    def Ixx(self) -> torch.Tensor: return torch.exp(self._log_Ixx)
    @property
    def Iyy(self) -> torch.Tensor: return torch.exp(self._log_Iyy)
    @property
    def Izz(self) -> torch.Tensor: return torch.exp(self._log_Izz)
    @property
    def k_thrust(self) -> torch.Tensor: return torch.exp(self._log_k_thrust)
    @property
    def k_torque(self) -> torch.Tensor: return torch.exp(self._log_k_torque)
    @property
    def arm_length(self) -> torch.Tensor: return torch.exp(self._log_arm_length)
    @property
    def tau_motor(self) -> torch.Tensor: return torch.exp(self._log_tau_motor)

    @classmethod
    def from_vehicle_params(
        cls, p: VehicleParams, dtype: torch.dtype = torch.float64
    ) -> "TorchVehicleParams":
        diag = p.inertia.diagonal()
        return cls(
            mass=float(p.mass),
            Ixx=float(diag[0]), Iyy=float(diag[1]), Izz=float(diag[2]),
            k_thrust=float(p.k_thrust), k_torque=float(p.k_torque),
            arm_length=float(p.arm_length), tau_motor=float(p.tau_motor),
            max_omega=float(p.max_omega), dtype=dtype,
        )

    def as_dict(self) -> dict[str, float]:
        """Snapshot of current scalar values for logging / comparison."""
        return {name: float(getattr(self, name).detach()) for name in _PARAM_NAMES}

    def named_scalars(self):
        """Iterate (name, current_scalar_value) for logging — uses property access."""
        for name in _PARAM_NAMES:
            yield name, getattr(self, name)
