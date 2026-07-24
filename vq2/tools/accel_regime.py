"""Rest-accel regime timeline: accel-implied pitch vs time per corpus,
annotated with race-clock resets and actuator activity.

Answers: WHEN does the IMU gravity vector read the canned -17.8 deg vs
level? (COR-147 CAM_TILT pair experiment, leg 1.)
"""
import json, math, struct, sys
import numpy as np

G_PITCH_CANNED = -17.8


def rows(path):
    imu, act, race = [], [], []
    for line in open(path):
        m = json.loads(line)
        t = m.get("_rx_wall")
        if m["mavpackettype"] == "HIGHRES_IMU":
            imu.append((t, m["xacc"], m["yacc"], m["zacc"]))
        elif m["mavpackettype"] == "ACTUATOR_OUTPUT_STATUS":
            act.append((t, max(m["actuator"][:4])))
        elif m["mavpackettype"] == "ENCAPSULATED_DATA":
            d = bytes.fromhex(m["data"])
            if len(d) >= 37 and d[0] == 1:
                f = struct.unpack_from("<BQqqIq", d, 0)
                race.append((t, f[1]))          # rx_wall, clock_ms
    return imu, act, race


def pitch_deg(fx, fy, fz):
    return math.degrees(math.atan2(fx, math.hypot(fy, fz)))


def timeline(name, path, bin_s=1.0):
    imu, act, race = rows(path)
    if not imu:
        print(f"{name}: no IMU")
        return
    t0 = imu[0][0]
    resets = [r[0] - t0 for i, r in enumerate(race[1:], 1)
              if race[i][1] < race[i - 1][1] - 5000]
    act_t = np.array([a[0] - t0 for a in act])
    act_v = np.array([a[1] for a in act])
    tt = np.array([r[0] - t0 for r in imu])
    ps = np.array([pitch_deg(r[1], r[2], r[3]) for r in imu])
    print(f"\n== {name}: {len(imu)} IMU rows over {tt[-1]:.1f}s, "
          f"resets at {[round(x,1) for x in resets]}")
    # bin: median pitch + actuator state per second
    prev_lab = None
    for lo in np.arange(0, tt[-1], bin_s):
        m = (tt >= lo) & (tt < lo + bin_s)
        if not m.any():
            continue
        p = float(np.median(ps[m]))
        am = (act_t >= lo) & (act_t < lo + bin_s)
        a = float(act_v[am].max()) if am.any() else float("nan")
        lab = ("CANNED" if abs(p - G_PITCH_CANNED) < 1.5 else
               "LEVEL" if abs(p) < 1.5 else f"{p:+.1f}")
        state = "thrust" if a == a and a > 0.2 else "rest"
        cur = (lab, state)
        if cur != prev_lab:                      # print transitions only
            print(f"  t={lo:6.1f}s  pitch {p:+7.2f}  [{lab:>6}]  {state}"
                  f"  act={a if a == a else -1:.2f}")
            prev_lab = cur
    # summary per regime
    for tag, m in [("rest", None)]:
        pass


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name = arg.rstrip("/\\").split("\\")[-1].split("/")[-1]
        timeline(name, arg + "/mavlink.jsonl")
