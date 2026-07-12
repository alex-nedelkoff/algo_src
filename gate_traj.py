"""Open gate trajectory + feedforward for the rate-interface tracker.

Generalizes the straight-cruise controller (race_cruise) to an arbitrary gate
layout. Geometry lives ONLY here (the planner); the controller that consumes a
`sample(s)` is trajectory-agnostic, so the SAME controller flies a
same-orientation "translated" course AND a gently curving corridor -- the only
difference is the gates fed in.

Design:
  * OPEN (non-looping) cubic spline through the gates -- natural BCs, no periodic
    wrap, so no return hairpin (the GateSpline reward-spline is closed).
  * True arc-length parameterization s (chord-knot u != arc length).
  * Analytic curvature from the spline derivatives (smooth) -- NOT instantaneous
    pure-pursuit curvature (noisy; that drove waypoint_nav's runaway).
  * Bank-limited speed schedule v(s)=min(v_cruise, sqrt(a_lat_max/kappa)). On a
    straight kappa->0 so v=v_cruise and the FF vanishes -> reduces to race_cruise.
  * Nose-follows-tangent heading (sideslip ~0 -> weathervane-neutral). For
    camera-forward racing, body_x points OPPOSITE travel, so yaw = atan2(-t).

Pure: numpy + scipy only, no sim deps -> testable offline.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

G = 9.81


class GateTrajectory:
    def __init__(self, gates, v_cruise: float = 2.5, phi_max_deg: float = 35.0,
                 samples_per_seg: int = 80, tilt_budget_deg: float = 25.0,
                 c_drag: float = 0.057, margin: float = 0.6, vz_max: float = 1e9):
        gates = np.asarray(gates, float)
        if len(gates) < 2:
            raise ValueError("need >= 2 gates")
        self.gates = gates
        self.v_cruise = float(v_cruise)
        # Controlled-sink: cap the DESCENT RATE (m/s) so a steep glideslope is flown at LOW forward
        # speed (drop altitude slowly, vq_course-style) instead of gliding down at cruise speed. On
        # the VQ1 descent, glideslope-at-speed overspeeds live (brake-vs-descend conflict, 06-15);
        # capping vertical speed forces v_fwd = vz_max/sin(slope) -> slow descent. 1e9 = off.
        self.vz_max = float(vz_max)
        self.a_lat_max = G * np.tan(np.radians(phi_max_deg))
        self.tilt_budget = G * np.tan(np.radians(tilt_budget_deg))   # horizontal accel budget
        self.c_drag = float(c_drag)                                  # REFIT-02 quadratic drag (/m)
        self.margin = float(margin)                                  # reserve for cross-track + gusts

        # chord-length knots, OPEN natural spline (no periodic wrap)
        seg = np.maximum(np.linalg.norm(np.diff(gates, axis=0), axis=1), 1e-6)
        u_knots = np.concatenate([[0.0], np.cumsum(seg)])
        self._sx = CubicSpline(u_knots, gates[:, 0], bc_type="natural")
        self._sy = CubicSpline(u_knots, gates[:, 1], bc_type="natural")
        self._sz = CubicSpline(u_knots, gates[:, 2], bc_type="natural")
        self._u_max = float(u_knots[-1])

        # dense grid -> true arc length s, and analytic curvature on the grid
        u = np.linspace(0.0, self._u_max, int(samples_per_seg * len(gates)))
        P = np.column_stack([self._sx(u), self._sy(u), self._sz(u)])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        d1 = np.column_stack([self._sx(u, 1), self._sy(u, 1), self._sz(u, 1)])
        d2 = np.column_stack([self._sx(u, 2), self._sy(u, 2), self._sz(u, 2)])
        cross = np.cross(d1, d2)
        denom = np.clip(np.linalg.norm(d1, axis=1) ** 3, 1e-9, None)
        self._u = u
        self._s = s
        self._P = P
        self.s_max = float(s[-1])
        self._kappa = np.linalg.norm(cross, axis=1) / denom        # unsigned 3D curvature
        self._kappa_signed = cross[:, 2] / denom                   # horizontal signed (yaw/bank dir)

    def _u_at_s(self, s: float) -> float:
        return float(np.interp(np.clip(s, 0.0, self.s_max), self._s, self._u))

    def nearest_s(self, pos) -> float:
        """Arc length of the trajectory sample nearest to pos (drone projection)."""
        d = self._P - np.asarray(pos, float)
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        return float(self._s[i])

    def nearest_s_window(self, pos, lo: float, hi: float) -> float:
        """nearest_s restricted to s in [lo, hi].

        For self-adjacent paths (legs passing within a few meters of each
        other) the global projection can leap between legs; windowing around
        the previous s keeps the carrot on the leg being flown.
        """
        m = (self._s >= max(0.0, lo)) & (self._s <= min(self.s_max, hi))
        if not m.any():
            return float(np.clip(lo, 0.0, self.s_max))
        d = self._P[m] - np.asarray(pos, float)
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        return float(self._s[m][i])

    def speed_at(self, kappa_abs: float) -> float:
        """Tilt-budget speed law: forward drag (c*v^2) + centripetal (v^2*kappa) <= budget*margin,
        solved for v. Straight (kappa->0) -> sqrt(budget*margin/c_drag) = the drag-saturation ceiling;
        curvature lowers it. Unifies the straight-line and lateral walls (REFIT-02 / LATERAL-WALL)."""
        denom = self.c_drag + max(float(kappa_abs), 0.0)
        v_budget = np.sqrt(self.tilt_budget * self.margin / denom)
        return float(min(self.v_cruise, v_budget))

    def sample(self, s: float) -> dict:
        """Reference at arc length s: pos(3 NED), tang(unit 3d), kappa(signed horiz),
        v(scheduled speed), yaw(heading of body_x), yaw_rate, a_lat(centripetal)."""
        u = self._u_at_s(s)
        pos = np.array([float(self._sx(u)), float(self._sy(u)), float(self._sz(u))])
        d1 = np.array([float(self._sx(u, 1)), float(self._sy(u, 1)), float(self._sz(u, 1))])
        tang = d1 / max(np.linalg.norm(d1), 1e-9)
        kabs = float(np.interp(u, self._u, self._kappa))
        ksig = float(np.interp(u, self._u, self._kappa_signed))
        v = self.speed_at(kabs)
        # controlled-sink descent-rate cap: on a descent (tang_z < 0), limit v so v*|tang_z| <= vz_max
        if tang[2] < -0.05:
            v = min(v, self.vz_max / abs(tang[2]))
        yaw = float(np.arctan2(-tang[1], -tang[0]))       # body_x opposite travel (camera-forward)
        return dict(pos=pos, tang=tang, kappa=ksig, v=v, yaw=yaw,
                    yaw_rate=v * ksig, a_lat=v * v * ksig)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    def show(name, gates, v=2.5):
        tr = GateTrajectory(gates, v_cruise=v)
        print(f"\n=== {name}: {len(gates)} gates, arc length {tr.s_max:.1f} m ===")
        print("   s_frac   pos(x,y,z)            v    bank   yaw    yawrate")
        for frac in np.linspace(0.0, 1.0, 11):
            r = tr.sample(frac * tr.s_max)
            bank = np.degrees(np.arctan2(abs(r["a_lat"]), G))
            print(f"   {frac:4.2f}    {r['pos']}   {r['v']:.2f}  {bank:4.1f}  "
                  f"{np.degrees(r['yaw']):+6.1f} {np.degrees(r['yaw_rate']):+6.1f}")

    # (A) same-orientation, translated: straight corridor, gentle lateral/vert jogs
    show("same-orientation translated",
         [[0, 0, -3], [12, 1.5, -3.3], [24, -1.0, -2.8], [36, 2.0, -3.5], [48, 0, -3]])
    # (B) gently curving corridor (course-1-like: marches + sweeps to ~one side)
    show("curving corridor",
         [[0, 0, -3], [12, 1, -3], [23, 4, -3.5], [33, 9, -3], [41, 16, -4]])
