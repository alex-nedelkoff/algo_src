import numpy as np
d=np.load("/tmp/vq_replay/run12_trueabort.npz")
def Rm(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def tt(q): q=q[[1,2,3,0]]; q=q/np.linalg.norm(q); return np.degrees(np.arccos(np.clip(Rm(q)[2,2],-1,1)))
q=d["quat"]; pos=d["pos"]; t=(d["t_us"]-d["t_us"][0])/1e6
true=np.array([tt(x) for x in q])
# gate0 ~ along course; use dist proxy = distance from origin grows in fly-away. Phase by time.
print("phase           frames  true_tilt med/max   omega med/max")
om=np.linalg.norm(d["omega"],axis=1)
for lab,(lo,hi) in [("approach t0-6",(0,6)),("near-gate t6-9",(6,9)),("flyaway t9+",(9,99))]:
    m=(t>=lo)&(t<hi)
    if m.sum()<3: continue
    print(f"  {lab:16s} {m.sum():4d}   {np.median(true[m]):4.0f}/{true[m].max():4.0f}        {np.median(om[m]):.1f}/{om[m].max():.1f}")
