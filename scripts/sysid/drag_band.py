"""Measure true D_eff(|v|) from IMU body-x specific force (acc_x = -(Dx+qx|v|)*vbx, drag-only
on body-x -- thrust is body-z, gravity not in IMU). Block means (kill dither). Compare to the
model's D_eff = 0.1 + 0.032|v|. Deploy + ramp recordings (the v5-20 approach band)."""
import numpy as np, glob, json
BLOCK=0.25
m=json.load(open("/Users/alex/Documents/drone-ai-grand-prix/algo_src/sysid/vq_model.json"))
Dx0=m["drag_linear_body"]["Dx"]; qx0=m["drag_quadratic_body"]["qx"]
runs=sorted(glob.glob("/tmp/vq_replay/2026061[02]*deploy4.npz"))+["/tmp/vq_replay/run_brk2.npz","/tmp/vq_replay/run12_trueabort.npz","/tmp/vq_replay/20260610T230939.npz"]
VB,AX,OM=[],[],[]
for p in runs:
    try: d=np.load(p)
    except: continue
    t=d["t_wall"]-d["t_wall"][0]; t0=t[0]; 
    while t0<t[-1]:
        msk=(t>=t0)&(t<t0+BLOCK)
        if msk.sum()>=3:
            VB.append(d["vel"][msk].mean(0)); AX.append(d["acc"][msk].mean(0)); OM.append(np.abs(d["omega"][msk]).mean(0))
        t0+=BLOCK
VB=np.array(VB); AX=np.array(AX); OM=np.array(OM)
vbx=VB[:,0]; vmag=np.linalg.norm(VB,axis=1); ax=AX[:,0]
ctrl=(OM.max(1)<2.0)&np.isfinite(ax)&(np.abs(vbx)>1.0)   # need vbx to divide
Deff_meas=-ax/vbx   # D_eff = -acc_x/vbx
print(f"blocks {len(VB)}, usable {ctrl.sum()}")
print(" |v| band   n   D_eff measured   D_eff model(0.1+0.032v)   ratio meas/model")
for lo,hi in [(2,5),(5,8),(8,12),(12,18),(18,26),(26,40)]:
    msk=ctrl&(vmag>=lo)&(vmag<hi)
    if msk.sum()<5: continue
    dm=np.median(Deff_meas[msk]); vmid=np.median(vmag[msk]); dmod=Dx0+qx0*vmid
    print(f"  {lo:2d}-{hi:2d}   {msk.sum():4d}     {dm:6.3f}          {dmod:6.3f}              {dm/dmod:.2f}")
