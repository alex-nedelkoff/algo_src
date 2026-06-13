"""Gate-0 approach geometry from a deploy recording: along-track vs lateral distance over time.
Tells if the 'closest 8.7m' miss is RANGE (overshoot past gate) or LATERAL (veer off aperture)."""
import numpy as np, sys
sys.path.insert(0,"/Users/alex/Documents/drone-ai-grand-prix/algo_src")
from scripts.sysid.closed_loop_policy import quat_to_R, to_enu, rotxy, wrap, quat_to_euler, qfix, B
d=np.load("/tmp/vq_replay/run12_trueabort.npz")
t=(d["t_us"]-d["t_us"][0])/1e6
# course exactly as vq_deploy4: anchor at spawn (row 0), camera axis, turn 0.2 space 14
q0=d["quat"][0]; pos0=d["pos"][0]*B
cf=-(quat_to_R(qfix(q0))[:,0])*B; cf[2]=0; cf/=np.linalg.norm(cf)+1e-9
hd=float(np.arctan2(cf[1],cf[0])); hd+=0.2; g0=pos0+14.0*np.array([np.cos(hd),np.sin(hd),0.0]); gyaw=hd
print("  t   dist  along(obs0)  lateral(obs1)  z    note")
mind=99; minrow=None
for k in range(0,len(t),max(1,int(0.3/(t[1]-t[0]) if len(t)>1 else 1))):
    pe=d["pos"][k]*B
    rel=rotxy((pe-g0)[:2], gyaw)   # gate-yaw frame: [along, lateral]
    dist=float(np.linalg.norm((g0-pe)[:2]))
    if dist<mind: mind=dist; minrow=(t[k],rel[0],rel[1],pe[2]-g0[2])
    if t[k]<10 and int(t[k]*2)%2==0:
        print(f"{t[k]:5.1f}  {dist:5.1f}   {rel[0]:+7.1f}     {rel[1]:+7.1f}   {pe[2]-g0[2]:+5.1f}")
print(f"\nCLOSEST: t={minrow[0]:.1f}  along={minrow[1]:+.1f}  lateral={minrow[2]:+.1f}  z={minrow[3]:+.1f}")
print("(along<0 = past the gate plane; |lateral| large = missed the aperture sideways)")
