@echo off
REM ============================================================================
REM Fast-DPVO observe-only flight: bounded per-frame latency.
REM   Item 1  half-res 320x180 + patches=24
REM   Item 2  bounded keyframe graph (REMOVAL_WINDOW=12, OPT_WINDOW=6)
REM Offline-measured plateau: 300ms -> 171ms sim-closed (~5.8Hz), ~4Hz live.
REM ONE variable per experiment: this differs from fly_servo_dpvo_observe.bat
REM ONLY in the DPVO input/graph knobs + its matching calibration.
REM Scale is NOT reproducible run-to-run (DPVO RANDOM centroids) -> OBSERVE ONLY.
REM ============================================================================
cd /d C:\Users\alexj
set VMAX=0.6
set POLICY=huber
set TICKS=2
set OBSZ=0
set G2TEST=1
set NOFIX=1
set RECENTER=0
set NOTRIM=1
set RELEVEL=1
set AIMBIAS_Y=0.0
set AIMBIAS_Z=0.3
set PUNCH_TRIG=4.5
set PUNCH_WIN=4.0
set PUNCH_LAT=1.2
set PUNCH_STRAIGHT=1
set PUNCH_VYBIAS=-0.3
set PUNCH_VX=2.2
set APPROACH_VYBIAS=-0.3
set CLIMB_S=0.9
set FASTGATE=1
set FGPURSUIT=1
set ATTMODE=1
set ATT_KP=0.6
set YAWCAL=0
set FGP_PUNCH_CTR=0.10
set RECENTER_Y=0.0
set RECENTER_Z=-0.4
set FGP_PLANE_X=0
set MAPFOLLOW=1
set MF_DR=1
set PN_PUNCH=1
set LINEFOLLOW=1
set GO_AROUND=1
set MF_STAGE_BACK=4.5
set MF_PITCH=0.09
set MF_YRMAX=0.3
set GN_UNLOAD=1
set GNSCALE=0
set DPVO=1
set DPVO_BRIDGE=1
set DPVO_ROUTE=1
set DPVO_OBSERVE=1
set DPVO_CAL=C:/Users/alexj/dpvo_fg62_fast.json
REM --- Item 1: half-res input + fewer patches ---
set DPVO_PATCHES=24
set DPVO_WIDTH=320
set DPVO_HEIGHT=180
REM --- Item 2: bounded keyframe graph ---
set DPVO_REMOVAL_WINDOW=12
set DPVO_OPT_WINDOW=6
set DPVO_STRIDE=2
set DPVO_CUDA_FRACTION=0.48
set DPVO_MAX_SPEED=8.0
set RRD=
set RECORD=C:/Users/alexj/vq2_servo_dpvo_fast
echo ==== fly_servo_dpvo_fast start %date% %time% ====
C:\Users\alexj\miniconda3\envs\monorace\python.exe -u C:\Users\alexj\vq2wp.py
echo ==== fly_servo_dpvo_fast exit %errorlevel% %time% ====
