"""fit_camcal.py -- offline joint camera calibration from camcal_samples.npz.

Two competing models for the measured -21.7 deg ray-azimuth offset:
  A (MOUNT yaw, rotates with body):  ray_w = H(alpha) @ R_t @ Rz(psi) @ R_opt(tau) @ d_opt
  B (WORLD/yaw-chain, fixed in world): ray_w = H(alpha) @ Rz(psi) @ R_t @ R_opt(tau) @ d_opt
Distinguishable because body yaw varies across the flight. Joint unknowns per model:
gate position g(3), psi (azimuth), tau (camera pitch), alpha (H reflection axis).
Objective: angular residual between ray_w_i and the unit vector (g - pos_i) -- bearing-only,
independent of the range/ratio model. Lower-residual model wins.
"""
import sys
import numpy as np
from scipy.optimize import least_squares

CX, FX, CY, FY = 320.0, 320.0, 180.0, 320.0


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qfix(q):
    q = np.asarray(q, float)
    return q[[1, 2, 3, 0]]


def r_opt(s_cam, tau):
    S, C = np.sin(tau), np.cos(tau)
    return np.array([[0.0, s_cam * S, s_cam * C],
                     [s_cam, 0.0, 0.0],
                     [0.0, C, -S]])


def Rz(p):
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Hrefl(alpha):
    c2, s2 = np.cos(2 * alpha), np.sin(2 * alpha)
    return np.array([[c2, s2, 0.0], [s2, -c2, 0.0], [0.0, 0.0, 1.0]])


def main(path):
    d = np.load(path)["raw"]
    print(f"{len(d)} raw detections")
    # columns: t,u,v,ring_w,ring_h,hole_w,hole_h,px,py,pz,qw,qx,qy,qz,tilt
    u, v = d[:, 1], d[:, 2]
    pos = d[:, 7:10]
    quat = d[:, 10:14]
    # filter: plausible detections only
    keep = (d[:, 3] > 25) & (d[:, 3] < 280)
    u, v, pos, quat = u[keep], v[keep], pos[keep], quat[keep]
    n = len(u)
    print(f"{n} after ring filter")
    d_opt = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones(n)], 1)
    d_opt /= np.linalg.norm(d_opt, axis=1, keepdims=True)
    R_ts = np.array([quat_to_R(qfix(q)) for q in quat])
    # s_cam: derive per canonical -- assume +1 (today's sessions all +1); flip via arg if needed
    s_cam = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    def rays(model, psi, tau, alpha):
        Ro = r_opt(s_cam, tau)
        H = Hrefl(alpha)
        out = np.empty((n, 3))
        if model == "A":
            M = Rz(psi) @ Ro
            for i in range(n):
                out[i] = H @ (R_ts[i] @ (M @ d_opt[i]))
        else:
            Rp = Rz(psi)
            for i in range(n):
                out[i] = H @ (Rp @ (R_ts[i] @ (Ro @ d_opt[i])))
        return out

    def resid(x, model):
        gx, gy, gz, psi, tau, alpha = x
        g = np.array([gx, gy, gz])
        rw = rays(model, psi, tau, alpha)
        tgt = g[None, :] - pos
        tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
        return (rw - tgt).ravel()

    # init: gate ~12 m along mean ray from mean pos, psi 0, tau 20deg, alpha 0
    r0 = rays("A", 0.0, np.radians(20.0), 0.0).mean(0)
    g0 = pos.mean(0) + 8.0 * r0 / np.linalg.norm(r0)
    x0 = np.array([g0[0], g0[1], g0[2], 0.0, np.radians(20.0), 0.0])
    for model in ("A", "B"):
        sol = least_squares(resid, x0, args=(model,), method="lm", max_nfev=2000)
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        gx, gy, gz, psi, tau, alpha = sol.x
        print(f"\nMODEL {model} ({'mount yaw' if model == 'A' else 'world/yaw-chain'}):")
        print(f"  residual RMS {rms:.4f} (unit-vector units; ~{np.degrees(rms):.2f} deg-ish)")
        print(f"  gate = [{gx:+.2f}, {gy:+.2f}, {gz:+.2f}]")
        print(f"  psi (azimuth) = {np.degrees(psi):+.2f} deg")
        print(f"  tau (cam pitch) = {np.degrees(tau):+.2f} deg")
        print(f"  alpha (H axis) = {np.degrees(alpha):+.2f} deg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "camcal_samples.npz")
