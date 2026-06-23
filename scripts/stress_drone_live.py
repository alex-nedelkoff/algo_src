"""LIVE adversarial stress of the Drone facade on the VQ sim (run on the laptop, sim up).

Hammers the threaded mission lifecycle + control against the real plant: preempt storm, pause/resume
mid-flight, abort+recover, orbit, stop-on-a-dime, multi-wp follow. A background watchdog tracks max
tilt/vel; the Commander wrapper guards every outgoing command for NaN/inf.  Usage:
  cd <repo root>; python scripts/stress_drone_live.py --rrd stress.rrd   (VQ sim must be up)
"""
import os, sys, time, threading
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import aigp.flight_telemetry as ftm
from aigp.commander import Commander
from aigp.drone import Drone, FlightConfig, Result
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import load_plant, _qfix
from aigp.state import Store

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
RES = []
def rec(ok, name, detail=""):
    RES.append((ok, name, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


class CountCmd:
    def __init__(self, real): self.real = real; self.n = 0; self.bad = 0; self.maxthr = 0.0
    def send_attitude_target(self, rate, thr):
        r = np.asarray(rate, float)
        if not (np.all(np.isfinite(r)) and np.isfinite(thr)): self.bad += 1
        self.n += 1; self.maxthr = max(self.maxthr, float(thr))
        return self.real.send_attitude_target(rate, thr)


class Watchdog(threading.Thread):
    """Poll drone state; track max tilt + max horizontal speed live, per phase."""
    def __init__(self, store):
        super().__init__(daemon=True); self.s = store; self.stop = False
        self.cur = "init"; self.byphase = {}; self.maxtilt = 0.0; self.maxv = 0.0
    def run(self):
        while not self.stop:
            ds = self.s.get_drone()
            if ds is not None:
                R = quat_to_R(_qfix(ds.quat_wxyz))   # TRUE frame (wire->true shuffle), matches navigator
                tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, R[2, 2])))))
                self.maxtilt = max(self.maxtilt, tilt)
                self.maxv = max(self.maxv, float(np.linalg.norm(ds.vel_ned[:2])))
                self.byphase[self.cur] = max(self.byphase.get(self.cur, 0.0), tilt)
            time.sleep(0.03)


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
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)): break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(m, boot); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0: return True
        time.sleep(0.02)
    return False


def main():
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10), "no heartbeat"; m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); real = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, real, m, boot), "sim not live after reset"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "stress_live", store=s)

    cc = CountCmd(real)
    wd = Watchdog(s); wd.start()
    drone = Drone(s, cc, plant, config=FlightConfig(), flog=flog)
    drone.set_origin(pos_ned=spawn, yaw=yaw0); real.arm()
    base_threads = threading.active_count()

    def settle(t=2.0):
        wd.cur = "settle"; drone.hover(seconds=t).wait(timeout=t + 5)

    # 1. PREEMPT STORM (live): 15 rapid retargets in a small box, each aborts the prior.
    print("\n=== 1. preempt storm (15 rapid gotos) ===", flush=True)
    wd.cur = "1 storm"
    box = [(2, 0, 0), (2, 2, 0), (0, 2, 0), (-2, 2, 0), (-2, 0, 0), (0, -2, 0)]
    ms = []
    for i in range(15):
        ms.append(drone.goto(box[i % len(box)], yaw="course")); time.sleep(0.15)
    fin = ms[-1].wait(timeout=30); time.sleep(0.3)
    leaked = threading.active_count() - base_threads
    priors_done = all(mm.done for mm in ms[:-1])
    rec(fin.result is Result.REACHED, "storm final reached", f"-> {fin.result.value}")
    rec(priors_done, "storm all priors done", f"-> {priors_done}")
    rec(leaked <= 1, "storm no thread leak", f"-> {leaked} extra threads")
    settle()

    # 2. PAUSE / RESUME mid-flight: drone must freeze in place while paused.
    print("\n=== 2. pause/resume mid-flight ===", flush=True)
    wd.cur = "2 pause"
    mm = drone.goto((9, 0, 0), yaw="course"); time.sleep(2.5)   # let it get moving (~cruise)
    freeze = s.get_drone().pos_ned.copy()
    mm.pause(); time.sleep(4.0)                                 # brake (DECEL_MAX) to a stop + settle
    pa = s.get_drone().pos_ned.copy(); time.sleep(0.8); pb_ = s.get_drone().pos_ned.copy()
    held = float(np.linalg.norm((pb_ - pa)[:2]))               # stationary now?
    ret = float(np.linalg.norm((pb_ - freeze)[:2]))            # distance from the freeze point
    rec(held < 0.6, "paused brakes + holds", f"-> stationary {held:.2f} m/0.8s; brake dist {ret:.1f} m (DECEL_MAX-limited)")
    mm.resume(); st = mm.wait(timeout=30)
    rec(st.result is Result.REACHED, "resume completes", f"-> {st.result.value}")
    settle()

    # 3. ABORT + RECOVER: abort mid-flight, then hover must catch + hold the drone.
    print("\n=== 3. abort + recover ===", flush=True)
    wd.cur = "3 abort"
    mm = drone.goto((7, 3, 0), yaw="course"); time.sleep(1.5)
    mm.abort(); ab = mm.wait(timeout=5)
    hv = drone.hover(seconds=2.5).wait(timeout=8)
    rec(ab.result is Result.ABORT, "abort reported", f"-> {ab.result.value}")
    rec(hv.result is Result.REACHED and hv.tilt_deg < 40, "recover hover", f"-> {hv.result.value} tilt={hv.tilt_deg:.0f}")
    settle()

    # 4. ORBIT (continuous): circle a point ahead.
    print("\n=== 4. orbit ===", flush=True)
    wd.cur = "4 orbit"
    st = drone.orbit((5, 0, 0), radius=3.0, seconds=8.0).wait(timeout=60)
    rec(st.result is Result.REACHED, "orbit completes", f"-> {st.result.value} tilt={st.tilt_deg:.0f}")
    settle()

    # 5. STOP-ON-A-DIME + multi-wp follow.
    print("\n=== 5. stop=True + follow ===", flush=True)
    wd.cur = "5 stop/follow"
    st = drone.goto((6, 0, 0), yaw="course", stop=True).wait(timeout=30)
    vstop = float(np.linalg.norm(s.get_drone().vel_ned[:2]))
    rec(st.result is Result.REACHED and vstop < 1.5, "stop-on-dime halts", f"-> {st.result.value} v_final={vstop:.2f}")
    st = drone.follow([(2, 2, 0), (4, 0, 0), (2, -2, 0)], yaw="course").wait(timeout=45)
    rec(st.result is Result.REACHED, "follow 3wp reached", f"-> {st.result.value}")
    settle()

    wd.stop = True
    rec(cc.bad == 0, "no NaN/inf command sent", f"-> {cc.bad} bad of {cc.n}")
    rec(getattr(wd, "maxtilt", 0) < 50, "max tilt bounded", f"-> peak tilt {getattr(wd,'maxtilt',0):.0f} deg")
    print(f"  peak horizontal speed: {getattr(wd,'maxv',0):.1f} m/s (envelope vmax=5)", flush=True)
    print("  per-phase peak tilt:", {k: round(v) for k, v in wd.byphase.items()}, flush=True)

    print("\n=== SUMMARY ===", flush=True)
    fails = [r for r in RES if not r[0]]
    print(f"  {len(RES) - len(fails)}/{len(RES)} pass", flush=True)
    for _, n, d in fails: print(f"  FAIL {n}: {d}", flush=True)
    time.sleep(1.5)


if __name__ == "__main__":
    main()
