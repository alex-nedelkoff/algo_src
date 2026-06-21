"""Live stress test of the Drone control facade against the VQ sim.

Conservative envelope (vmax 4, default_speed 3 -> inside the known-stable ~v5 live window) so we
test the API's lifecycle/control, not the high-speed plant runaway. Always-commanded sequence:
takeoff -> auto-abort+recover -> pause/resume -> preempt storm -> orbit -> land. Each mission is
monitored for max tilt / max speed / non-finite telemetry. Streams to the Mac dashboard by default.

Run on the laptop (sim up):  python scripts/live_stress.py
"""
import os, sys, time
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, os.getcwd())

import aigp.flight_telemetry as ftm
from aigp.commander import Commander
from aigp.drone import Drone, FlightConfig, Result
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import load_plant
from aigp.state import Store

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def idle(m, boot):
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start(s, c, m, boot):
    t = time.time()
    while time.time() - t < 1.0:
        idle(m, boot); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(m, boot); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(m, boot); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


REPORT = []


def monitor(s, m, label, dur):
    """Sample mission status + raw telemetry until done/timeout. Returns a result dict."""
    t0 = time.time(); mt = 0.0; mv = 0.0; n = 0; nonfinite = 0; minz = 1e9; maxz = -1e9
    while not m.done and time.time() - t0 < dur:
        st = m.status()
        mt = max(mt, st.tilt_deg); mv = max(mv, st.vel); n += 1
        d = s.get_drone()
        if d is not None:
            if not np.all(np.isfinite(d.pos_ned)) or not np.all(np.isfinite(d.vel_ned)):
                nonfinite += 1
            minz = min(minz, float(d.pos_ned[2])); maxz = max(maxz, float(d.pos_ned[2]))
        time.sleep(0.05)
    st = m.status()
    d = s.get_drone()
    pos = None if d is None else np.round(d.pos_ned, 2)
    row = dict(label=label, result=st.result.value, maxtilt=round(mt, 1), maxvel=round(mv, 1),
               nonfinite=nonfinite, z_range=(round(minz, 2), round(maxz, 2)), end_pos=pos,
               gate=(s.get_race() or {}).get("active_gate_index"))
    REPORT.append(row)
    print(f"[{label}] result={row['result']} maxtilt={row['maxtilt']} maxvel={row['maxvel']} "
          f"nonfinite={nonfinite} z=[{row['z_range'][0]},{row['z_range'][1]}] end={pos}", flush=True)
    return st


def main():
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10), "no heartbeat"; m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "live_stress", store=s)

    cfg = FlightConfig(vmax=4.0, amax=3.5, tilt_deg=18.0, default_speed=3.0)
    drone = Drone(s, c, plant, config=cfg, flog=flog)
    drone.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()
    print(f"spawn={np.round(spawn,2)} yaw0={yaw0:.2f}", flush=True)

    # T1: takeoff 2 m (blocking)
    monitor(s, drone.takeoff(2.0), "T1 takeoff", 10); drone._mission.wait(timeout=10)

    # T2: auto-abort + recover. goto, then hover() preempts it (auto-abort) and HOLDS (no fall).
    gm = drone.goto((8, 0, 0), yaw="course")
    monitor(s, gm, "T2a goto(running)", 1.2)
    hm = drone.hover(2.0)                      # preempts gm -> should auto-ABORT gm, then hold
    aborted = gm.result is Result.ABORT
    monitor(s, hm, "T2b hover-recover", 3); hm.wait(timeout=3)
    print(f"   T2 auto-abort-of-prior={aborted}", flush=True)

    # T3: pause / resume mid-flight (pause loop actively holds, no fall)
    gm = drone.goto((6, 5, 0), yaw="course")
    monitor(s, gm, "T3a goto(running)", 1.2)
    gm.pause(); time.sleep(1.5)
    p_paused = gm.status().progress
    time.sleep(0.8)
    frozen = abs(gm.status().progress - p_paused) < 1e-6
    gm.resume()
    monitor(s, gm, "T3b after-resume", 8); gm.wait(timeout=8)
    print(f"   T3 progress_frozen_while_paused={frozen}", flush=True)

    # T4: preempt storm — 6 rapid gotos; only the last should complete, controlled throughout
    pts = [(6, 0, 0), (2, 6, 0), (8, 4, 0), (0, 6, 0), (8, 0, 0), (3, 3, 0)]
    last = None
    for i, p in enumerate(pts):
        last = drone.goto(p, yaw="course")
        monitor(s, last, f"T4 storm[{i}]", 0.4)
    monitor(s, last, "T4 storm-final", 10); last.wait(timeout=10)

    # T5: orbit a point, camera locked
    monitor(s, drone.orbit((5, 0, 0), radius=4.0, seconds=6.0), "T5 orbit", 9)
    drone._mission.wait(timeout=9)

    # T6: land
    monitor(s, drone.land(2.5), "T6 land", 10); drone._mission.wait(timeout=10)

    print("\n=== LIVE STRESS REPORT ===", flush=True)
    bad = [r for r in REPORT if r["nonfinite"] or r["maxtilt"] > 50 or r["result"] == "error"]
    for r in REPORT:
        print(f"  {r['label']:20s} {r['result']:8s} tilt<={r['maxtilt']:4} v<={r['maxvel']:4} "
              f"nf={r['nonfinite']} z={r['z_range']}", flush=True)
    print(f"\nFLAGS: {len(bad)} concerning rows" + (f" -> {[r['label'] for r in bad]}" if bad else ""))
    print("LIVE_STRESS_DONE", flush=True)


if __name__ == "__main__":
    main()
