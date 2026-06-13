"""Measure the live ring frequency + damping per axis from the omega oscillation in the
turn-in/flee segments of run-9 and run-10 (where the ring fires). FFT peak + log-decrement."""
import numpy as np
SGN = np.array([1.0,-1.0,1.0]); DT=1/72
for run in ("run9","run10"):
    d = np.load(f"/tmp/vq_replay/{run}.npz")
    t = (d["t_us"]-d["t_us"][0])/1e6
    _,u=np.unique(d["t_us"],return_index=True); u=np.sort(u)
    om = d["omega"][u]*SGN; tt=t[u]
    # ring segment: t 3-9s (POL phase with the turn-in ring)
    seg = (tt>=3.0)&(tt<=9.0)
    print(f"\n{run}: ring segment {seg.sum()} samples")
    for ax,nm in ((0,"roll"),(1,"pitch"),(2,"yaw")):
        x = om[seg,ax]-np.mean(om[seg,ax])
        if len(x)<32: continue
        f = np.fft.rfftfreq(len(x), DT); P=np.abs(np.fft.rfft(x))**2
        # peak above 2 Hz (ignore DC/maneuver band)
        mask = f>2.0
        pk = f[mask][np.argmax(P[mask])]
        # crude damping: ratio of successive peak envelopes via Hilbert-free local maxima
        print(f"  {nm}: FFT peak {pk:.1f} Hz  rms {np.std(x):.2f}")
