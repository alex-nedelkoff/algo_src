"""Visual servo: gate bearing -> heading / altitude / forward setpoints. Pure.
Camera intrinsics fx=fy=320, cx=320, cy=180 (aigp.geometry.K). Pixel->control signs
(sign_x, sign_y) are pinned empirically in bring-up (servo_sign_probe.py)."""
from __future__ import annotations
from dataclasses import dataclass

FX = FY = 320.0
CX = 320.0
CY = 180.0


@dataclass
class ServoCfg:
    k_yaw: float = 0.8       # rad per unit ex (horizontal bearing)
    k_alt: float = 6.0       # m per unit ey (z-setpoint nudge)
    fwd_speed: float = 1.8   # m/s along camera heading (capped, weathervane-safe)
    sign_x: float = 1.0      # pinned in bring-up
    sign_y: float = 1.0
    max_dz: float = 8.0      # clamp altitude nudge
    coast_frac: float = 0.5  # forward-speed fraction when gate momentarily lost


@dataclass
class ServoCmd:
    fwd_speed: float
    yaw_sp: float
    z_sp: float
    have_gate: bool


def servo(det, yaw_cur: float, z_cur: float, cfg: ServoCfg, last: ServoCmd | None = None) -> ServoCmd:
    if det is None:
        if last is not None:
            return ServoCmd(cfg.fwd_speed * cfg.coast_frac, last.yaw_sp, last.z_sp, False)
        return ServoCmd(0.0, yaw_cur, z_cur, False)
    ex = (det.u - CX) / FX
    ey = (det.v - CY) / FY
    yaw_sp = yaw_cur + cfg.k_yaw * cfg.sign_x * ex
    dz = cfg.k_alt * cfg.sign_y * ey
    dz = max(-cfg.max_dz, min(cfg.max_dz, dz))
    return ServoCmd(cfg.fwd_speed, yaw_sp, z_cur + dz, True)
