"""Re-examine RL deploy runs with TRUE (qfix) tilt vs the warped live-frame tilt we aborted on."""
import numpy as np
def Rm(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def tilt(q): return np.degrees(np.arccos(np.clip(Rm(q)[2,2],-1,1)))
for name,f in [("run9 q92","run9"),("run10 brake","run10"),("run11 slew","run11_slew")]:
    d=np.load(f"/tmp/vq_replay/{f}.npz"); q=d["quat"]
    live=np.array([tilt(ql) for ql in q])
    true=np.array([tilt(ql[[1,2,3,0]]/np.linalg.norm(ql[[1,2,3,0]])) for ql in q])
    om=np.linalg.norm(d["omega"],axis=1)
    print(f"\n{name}: {len(q)} frames")
    print(f"  LIVE-frame tilt: max {live.max():.0f}  med {np.median(live):.0f}  >90deg {100*(live>90).mean():.0f}% of frames")
    print(f"  TRUE  -frame tilt: max {true.max():.0f}  med {np.median(true):.0f}  >60deg {100*(true>60).mean():.0f}% of frames")
    print(f"  omega: max {om.max():.1f}  med {np.median(om):.1f} rad/s")
