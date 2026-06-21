"""Adversarial stress test of the Drone control facade (no sim required).

Probes API contracts the unit tests don't: bad inputs (NaN/inf/empty/shape), error
surfacing (sync raise vs swallowed-to-ERROR), threading lifecycle (preempt storms,
double-abort, thread leaks), and look-at/origin preconditions. Uses the same fakes as
tests/test_aigp/test_drone.py. Run:  python scripts/stress_drone_api.py
"""
import os, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigp.drone import Drone, FlightConfig, Result, Mission
from aigp.state import DroneState

RES = []  # (severity, name, detail)
def rec(sev, name, detail=""): RES.append((sev, name, detail)); print(f"[{sev}] {name} {detail}")


class FakeStore:
    def __init__(self, ds): self._ds = ds
    def get_drone(self): return self._ds
    def get_race_live(self): return True


class RecCmd:
    def __init__(self): self.n = 0
    def send_attitude_target(self, rate, thr): self.n += 1


def mk(config=None):
    ds = DroneState(np.array([0, 0, -2.0]), np.zeros(3), np.array([0.0, 1, 0, 0]), np.zeros(3), 0)
    return Drone(FakeStore(ds), RecCmd(), (0.5, 13.0, np.array([1.0, 1.0, 1.0])),
                 config=config or FlightConfig())


def expect_raise(name, fn):
    try:
        fn(); rec("BUG", name, "expected ValueError, none raised")
    except ValueError as e:
        rec("OK", name, f"-> ValueError {str(e)[:50]!r}")
    except Exception as e:
        rec("WARN", name, f"-> {type(e).__name__} (not ValueError) {str(e)[:50]!r}")


# ---------------------------------------------------------------------------
# 1. Input validation — the facade should reject garbage BEFORE launching a thread
# ---------------------------------------------------------------------------
print("\n=== 1. INPUT VALIDATION ===")
d = mk(); d.set_origin(pos_ned=np.zeros(3), yaw=0.0)
d.nav.goto = lambda *a, **k: "reached"; d.nav.follow = lambda *a, **k: "reached"

expect_raise("goto bad frame", lambda: d.goto((1, 0, 0), frame="polar"))
expect_raise("goto bad yaw", lambda: d.goto((1, 0, 0), yaw="spin"))
expect_raise("goto speed=0", lambda: d.goto((1, 0, 0), speed=0))
expect_raise("goto speed<0", lambda: d.goto((1, 0, 0), speed=-3))
expect_raise("goto speed=NaN", lambda: d.goto((1, 0, 0), speed=float("nan")))
expect_raise("goto speed=inf", lambda: d.goto((1, 0, 0), speed=float("inf")))
expect_raise("orbit radius=inf", lambda: d.orbit((1, 0, 0), radius=float("inf"), seconds=1))
expect_raise("orbit seconds=inf", lambda: d.orbit((1, 0, 0), radius=2, seconds=float("inf")))
expect_raise("takeoff alt=inf", lambda: d.takeoff(float("inf")))
expect_raise("takeoff alt=NaN", lambda: d.takeoff(float("nan")))
expect_raise("takeoff alt=True(bool)", lambda: d.takeoff(True))

# NaN / inf / wrong-shape coordinates — does anything reject them?
for label, wp in [("NaN coord", (float("nan"), 0, 0)),
                  ("inf coord", (float("inf"), 0, 0)),
                  ("2D coord", (1, 0)),
                  ("4D coord", (1, 0, 0, 0)),
                  ("empty coord", ())]:
    try:
        m = d.goto(wp, yaw="hold"); s = m.wait(timeout=1)
        if s.result is Result.ERROR:
            rec("WARN", f"goto {label}", f"-> swallowed to ERROR in thread: {s.error}")
        else:
            rec("BUG", f"goto {label}", f"-> accepted silently, result={s.result.value} (no coord validation)")
    except Exception as e:
        rec("OK", f"goto {label}", f"-> raised {type(e).__name__}")

# empty follow list
m = d.goto  # noqa
try:
    s = d.follow([], yaw="hold").wait(timeout=2)
    if s.result is Result.REACHED:
        rec("WARN", "follow([])", "-> REACHED (no-op); arguably should ValueError")
    elif s.result is Result.ERROR:
        rec("BUG", "follow([])", f"-> ERROR in worker thread (unvalidated): {s.error}")
    else:
        rec("WARN", "follow([])", f"-> {s.result.value}")
except Exception as e:
    rec("OK", "follow([])", f"-> raised {type(e).__name__}")


# ---------------------------------------------------------------------------
# 2. Error surfacing consistency: origin-not-set
# ---------------------------------------------------------------------------
print("\n=== 2. ERROR SURFACING (origin not set) ===")
d2 = mk()  # no set_origin
try:
    s = d2.goto((1, 0, 0), yaw="hold").wait(timeout=2)
    rec("WARN", "goto no-origin", f"-> swallowed to {s.result.value} in thread: {s.error}")
except Exception as e:
    rec("OK", "goto no-origin", f"-> raised {type(e).__name__} synchronously")
d3 = mk()
try:
    d3.orbit((1, 0, 0), radius=2, seconds=1)
    rec("WARN", "orbit no-origin", "-> no error (built ring against missing origin)")
except Exception as e:
    rec("INFO", "orbit no-origin", f"-> raised {type(e).__name__} synchronously (inconsistent w/ goto-in-thread)")


# ---------------------------------------------------------------------------
# 3. look_at preconditions
# ---------------------------------------------------------------------------
print("\n=== 3. LOOK-AT PRECONDITIONS ===")
d4 = mk(); d4.set_origin(pos_ned=np.zeros(3), yaw=0.0); d4.nav.goto = lambda *a, **k: "reached"
expect_raise("yaw=lookat w/o point", lambda: d4.goto((1, 0, 0), yaw="lookat"))
try:
    d4.goto((1, 0, 0), yaw="lookat", look_at=(5, 0, 0)).wait(timeout=1)
    rec("OK", "yaw=lookat w/ look_at", "-> accepted")
except Exception as e:
    rec("BUG", "yaw=lookat w/ look_at", f"-> {type(e).__name__} {e}")


# ---------------------------------------------------------------------------
# 4. Threading lifecycle
# ---------------------------------------------------------------------------
print("\n=== 4. THREADING LIFECYCLE ===")

def slow_loop(d, order, tag):
    def fn(*a, **k):
        for _ in range(400):
            if d.nav._abort_evt is not None and d.nav._abort_evt.is_set():
                order.append(f"{tag}:abort"); return "abort"
            time.sleep(0.005)
        order.append(f"{tag}:finish"); return "reached"
    return fn

# 4a. preempt storm — 40 rapid gotos; each must abort the prior. No thread leak.
d5 = mk(); d5.set_origin(pos_ned=np.zeros(3), yaw=0.0)
order = []
base_threads = threading.active_count()
d5.nav.goto = slow_loop(d5, order, "g")
ms = []
for i in range(40):
    ms.append(d5.goto((i, 0, 0), yaw="hold")); time.sleep(0.002)
final = ms[-1].wait(timeout=3)
time.sleep(0.3)
leaked = threading.active_count() - base_threads
aborts = sum(1 for o in order if o.endswith(":abort"))
rec("OK" if final.result is Result.REACHED else "BUG", "preempt storm final",
    f"-> final result={final.result.value}, {aborts} prior missions aborted")
rec("OK" if leaked <= 1 else "BUG", "preempt storm thread leak",
    f"-> {leaked} extra live threads after settle (want ~0)")
prior_done = all(m.done for m in ms[:-1])
rec("OK" if prior_done else "BUG", "preempt storm all prior done", f"-> all_done={prior_done}")

# 4b. double-abort + abort-after-done are no-ops, not crashes
d6 = mk(); d6.set_origin(pos_ned=np.zeros(3), yaw=0.0); d6.nav.goto = slow_loop(d6, [], "x")
m = d6.goto((1, 0, 0), yaw="hold"); time.sleep(0.02)
try:
    m.abort(); m.abort(); m.wait(timeout=2); m.abort()
    rec("OK", "double + post-done abort", f"-> no crash, result={m.result.value}")
except Exception as e:
    rec("BUG", "double + post-done abort", f"-> {type(e).__name__} {e}")

# 4c. wait timeout while still running returns a snapshot (does not hang)
d7 = mk(); d7.set_origin(pos_ned=np.zeros(3), yaw=0.0); d7.nav.goto = slow_loop(d7, [], "w")
m = d7.goto((1, 0, 0), yaw="hold")
t0 = time.time(); s = m.wait(timeout=0.1); dt = time.time() - t0
ok = (not m.done) and dt < 0.5
rec("OK" if ok else "BUG", "wait(timeout) non-blocking", f"-> returned in {dt:.2f}s, done={m.done}, result={s.result.value}")
m.abort()

# 4d. run_fn raises -> ERROR with repr, thread survives
d8 = mk(); d8.set_origin(pos_ned=np.zeros(3), yaw=0.0)
def boom(*a, **k): raise RuntimeError("synthetic loop crash")
d8.nav.goto = boom
s = d8.goto((1, 0, 0), yaw="hold").wait(timeout=2)
ok = s.result is Result.ERROR and s.error and "synthetic" in s.error
rec("OK" if ok else "BUG", "loop crash -> ERROR", f"-> result={s.result.value}, error={s.error}")

# 4e. hover(None) is indefinite but abortable; a new motion call preempts it
d9 = mk(); d9.set_origin(pos_ned=np.zeros(3), yaw=0.0)
d9.nav.settle = lambda *a, **k: time.sleep(0.01)
d9.nav.goto = lambda *a, **k: "reached"
hm = d9.hover()  # indefinite
time.sleep(0.05)
running = not hm.done
gm = d9.goto((1, 0, 0), yaw="hold"); gm.wait(timeout=2); time.sleep(0.1)
rec("OK" if running and hm.done else "BUG", "hover(None) preempted by goto",
    f"-> hover_was_running={running}, hover_done_after_preempt={hm.done}")

print("\n=== SUMMARY ===")
from collections import Counter
c = Counter(s for s, _, _ in RES)
for sev in ("BUG", "WARN", "INFO", "OK"):
    if c[sev]:
        print(f"  {sev}: {c[sev]}")
bugs = [r for r in RES if r[0] == "BUG"]
if bugs:
    print("\nBUGS:")
    for _, n, det in bugs:
        print(f"  - {n}: {det}")
