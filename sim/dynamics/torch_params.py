"""Learnable PyTorch quadrotor parameters for system identification.

Mirrors the 8 scalar parameters that drive ``NumpyQuadDynamics.step()``:

    mass        : kg
    Ixx, Iyy, Izz : kg * m^2 (diagonal inertia tensor)
    k_thrust    : N / (rad/s)^2
    k_torque    : N * m / (rad/s)^2
    arm_length  : m
    tau_motor   : s (first-order motor time constant)

All scalars are stored as ``nn.Parameter`` so autograd tracks them.
``drag_coeff`` and ``max_rpm`` from the numpy ``VehicleParams`` are deliberately
held fixed for the grey-box MVP — drag is assumed zero and ``max_rpm`` is a hard
clip used only at the dynamics boundary, not a learnable scalar.
"""
from __future__ import annotations

import torch
from torch import nn

from sim.dynamics.params import VehicleParams


class TorchVehicleParams(nn.Module):
    """Learnable container for the 8 scalar parameters of the grey-box model."""

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
        self.mass = nn.Parameter(torch.tensor(mass, dtype=dtype))
        self.Ixx = nn.Parameter(torch.tensor(Ixx, dtype=dtype))
        self.Iyy = nn.Parameter(torch.tensor(Iyy, dtype=dtype))
        self.Izz = nn.Parameter(torch.tensor(Izz, dtype=dtype))
        self.k_thrust = nn.Parameter(torch.tensor(k_thrust, dtype=dtype))
        self.k_torque = nn.Parameter(torch.tensor(k_torque, dtype=dtype))
        self.arm_length = nn.Parameter(torch.tensor(arm_length, dtype=dtype))
        self.tau_motor = nn.Parameter(torch.tensor(tau_motor, dtype=dtype))
        # Fixed (not learned): used only as the action clip ceiling.
        self.register_buffer("max_omega", torch.tensor(max_omega, dtype=dtype))

    @classmethod
    def from_vehicle_params(
        cls, p: VehicleParams, dtype: torch.dtype = torch.float64
    ) -> "TorchVehicleParams":
        """Build a TorchVehicleParams identical to a VehicleParams snapshot."""
        diag = p.inertia.diagonal()
        return cls(
            mass=float(p.mass),
            Ixx=float(diag[0]),
            Iyy=float(diag[1]),
            Izz=float(diag[2]),
            k_thrust=float(p.k_thrust),
            k_torque=float(p.k_torque),
            arm_length=float(p.arm_length),
            tau_motor=float(p.tau_motor),
            max_omega=float(p.max_omega),
            dtype=dtype,
        )

    def as_dict(self) -> dict[str, float]:
        """Snapshot of current scalar values for logging / comparison."""
        return {
            "mass": float(self.mass.detach()),
            "Ixx": float(self.Ixx.detach()),
            "Iyy": float(self.Iyy.detach()),
            "Izz": float(self.Izz.detach()),
            "k_thrust": float(self.k_thrust.detach()),
            "k_torque": float(self.k_torque.detach()),
            "arm_length": float(self.arm_length.detach()),
            "tau_motor": float(self.tau_motor.detach()),
        }
