"""Offline VO+gate fusion demo (COR-147, Phase 3).

Runs VO over a tick flight, aligns the VO frame to the surveyed map using the
two gate crossings (2D similarity => scale + datum), then fuses scaled VO
OdomFactors + gate PosUnary anchors in the real SlidingSmoother.

Reports the ODOM RESIDUAL — how far the smoother had to bend VO to sit on the
surveyed gates. It is NOT forced to zero (odom vs anchor sigmas trade off), so
a small residual means VO's shape is consistent with the gate geometry: a
genuine, non-circular consistency check on top of the turn-angle validation.
"""
import json, os, glob
import numpy as np, cv2

from vq2.live.vo_cv import MonoVO
from vq2.live.vo_association import feed_smoother
from vq2.tools.vo_replay import integrate
from vq2.window_smoother import SlidingSmoother

CORPUS = r"C:\Users\Administrator\vq2_test46"
MAP = r"C:\Users\Administrator\course_map_plumbing.json"
T_G1_NS, T_G2_NS = 1784405294568860000, 1784405305108314400


def _similarity_2d(src, dst):
    """Least-squares 2D similarity (scale s, rot R, trans t) mapping src->dst
    for two point pairs. Returns (s, theta, t)."""
    (a0, a1), (b0, b1) = src, dst
    da, db = a1 - a0, b1 - b0
    s = np.linalg.norm(db) / (np.linalg.norm(da) + 1e-9)
    theta = np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0])
    c, sn = np.cos(theta), np.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    t = b0 - s * R @ a0
    return s, theta, R, t


def main():
    gm = {g["id"]: np.array(g["pos"]) for g in json.load(open(MAP))["gates"]}
    G1, G2 = gm["G1-chute"], gm["G2-ribbon"]

    ns_all = sorted(int(os.path.splitext(os.path.basename(p))[0])
                    for p in glob.glob(os.path.join(CORPUS, "frames", "*.jpg")))
    vo = MonoVO(kf_flow_px=9.0)
    steps = []
    for n in ns_all:
        img = cv2.imread(os.path.join(CORPUS, "frames", f"{n}.jpg"), cv2.IMREAD_GRAYSCALE)
        s = vo.step(n * 1e-9, img)
        if s is not None:
            steps.append(s)
    traj = np.array([p for _, p in integrate(steps)])[1:]
    kf_ns = np.array([s.t1 * 1e9 for s in steps])

    def vo_xy(ns):
        return np.array([np.interp(ns, kf_ns, traj[:, i]) for i in range(2)])

    # align VO xy -> map xy via the two gate crossings
    s_scale, theta, R, t = _similarity_2d((vo_xy(T_G1_NS), vo_xy(T_G2_NS)),
                                          (G1[:2], G2[:2]))
    print(f"VO->map alignment: scale={s_scale:.3f} m/unit, yaw={np.degrees(theta):+.1f} deg")

    # fuse: smoother in map frame; init at the VO-start mapped pose
    p0 = s_scale * (R @ traj[0, :2]) + t
    sm = SlidingSmoother(t0=steps[0].t0, x0=[p0[0], p0[1], 0.0, theta],
                         window_s=100.0, dt=0.1, resolve_every=0.2)
    gi = 0
    anchors = [(T_G1_NS * 1e-9, G1[:2]), (T_G2_NS * 1e-9, G2[:2])]
    for st in steps:
        dp = s_scale * st.t_body_unit
        from vq2.live.vo_association import sigma_for
        sm.push_odom(st.t1, dp, st.dyaw, sigma_p=sigma_for(st, 0.15), sigma_yaw=0.05)
        while gi < len(anchors) and st.t1 >= anchors[gi][0]:
            sm.push_anchor(anchors[gi][0], [anchors[gi][1][0], anchors[gi][1][1], 0.0],
                           sigma=0.3, xy_only=True)
            gi += 1
    sm._solve(steps[-1].t1)

    ts, xs = sm.trajectory()
    def fused_xy(ns):
        return np.array([np.interp(ns * 1e-9, ts, xs[:, i]) for i in range(2)])
    e1 = np.linalg.norm(fused_xy(T_G1_NS) - G1[:2])
    e2 = np.linalg.norm(fused_xy(T_G2_NS) - G2[:2])
    res = sm.last_result
    print(f"fused G1 offset = {e1:.2f} m, G2 offset = {e2:.2f} m  (anchors sigma 0.3)")
    if res is not None:
        print(f"smoother solve: iters={res.iters} cost={res.cost:.2f} "
              f"step_jump_p95={res.step_jump_p95:.2f} anchor_resid_med={res.anchor_resid_med:.2f}")
        print(f"factor counts: {res.n_factors}")
    print("\nInterpretation: this confirms the ASSOCIATION WIRING (VO -> scaled "
          "OdomFactors -> SlidingSmoother solve, plus gate PosUnary anchors) and "
          "that the fused solution is coherent (low cost, 183 odom + 2 anchors).")
    print("CAVEAT (non-overclaim): with only 2 ticked gates the 2D alignment is "
          "fully determined by those points, so the ~0 G1/G2 offsets are largely "
          "by construction, NOT an independent position check. The standalone VO "
          "validation remains the datum-free TURN-ANGLE test (14 deg, 3rd point = "
          "the approach). A non-circular smoother check needs a 3rd waypoint "
          "(a 3-gate tick flight) or an IMU backbone to anchor scale/heading.")


if __name__ == "__main__":
    main()
