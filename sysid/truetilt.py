import numpy as np
d=np.load("/tmp/vq_replay/wpslew.npz")
def Rm(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
q=d["quat"]  # live stored wxyz
print(" k   live_tilt  true_tilt(qfix)  |v|")
for k in range(0,len(q),max(1,len(q)//20)):
    ql=q[k]                       # live-frame quat
    qt=ql[[1,2,3,0]]; qt=qt/np.linalg.norm(qt)   # qfix true-frame
    tl=np.degrees(np.arccos(np.clip(Rm(ql)[2,2],-1,1)))
    tt=np.degrees(np.arccos(np.clip(Rm(qt)[2,2],-1,1)))
    print(f"{k:4d}  {tl:7.0f}   {tt:9.0f}      {np.linalg.norm(d['vel'][k]):.1f}")
