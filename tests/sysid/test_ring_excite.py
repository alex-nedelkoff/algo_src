import numpy as np
from scripts.sysid.ring_excite import excite_schedule

def test_schedule_shape_axis_and_ramp():
    dt = 1/720
    sched = excite_schedule(axis=0, dt=dt, tilt_setpoints=[20.0, 40.0],
                            chirp_f0=1.0, chirp_f1=12.0, seg_s=2.0, amp=3.0)
    assert sched.shape[1] == 4
    assert sched.shape[0] == int(2.0 / dt) * 2          # two segments
    assert np.any(sched[:, 0] != 0)                      # roll axis is driven
    assert np.allclose(sched[:, 2], 0.0)                # ...yaw is not
    assert abs(sched[0, 0]) < abs(sched[int(0.5/dt), 0]) # ramp-in: amplitude grows
    assert abs(sched[:, 0]).max() <= 3.0 + 1e-6          # respects amp cap
    assert sched[int(1.0/dt), 3] == 20.0                 # hold-tilt of segment 0
