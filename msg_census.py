"""msg_census.py -- 30 s census of EVERY MAVLink message type + ENCAP dtype the sim sends.
Finds message types we never parse (candidate scoring/pass channels)."""
import time, struct
from collections import Counter
from pymavlink import mavutil
import os
os.environ.setdefault("MAVLINK20", "1")

conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550", source_system=255)
conn.wait_heartbeat(timeout=10)
print("heartbeat ok; listening 30 s...", flush=True)
types = Counter(); encap = Counter(); samples = {}
t0 = time.time()
while time.time() - t0 < 30:
    m = conn.recv_match(blocking=True, timeout=1)
    if m is None:
        continue
    t = m.get_type()
    types[t] += 1
    if t not in samples and t not in ("BAD_DATA",):
        samples[t] = m.to_dict()
    if t == "ENCAPSULATED_DATA":
        raw = bytes(m.data)
        if raw:
            encap[raw[0]] += 1
print("\n--- message types (30 s) ---", flush=True)
for k, v in types.most_common():
    print(f"{k:32s} {v:6d}", flush=True)
print("\n--- ENCAP dtypes (first byte) ---", flush=True)
for k, v in encap.most_common():
    print(f"dtype {k:4d} {v:6d}", flush=True)
KNOWN = {"HEARTBEAT", "ODOMETRY", "HIGHRES_IMU", "RAW_IMU", "SCALED_IMU", "ATTITUDE",
         "ACTUATOR_OUTPUT_STATUS", "COLLISION", "DATA_TRANSMISSION_HANDSHAKE",
         "ENCAPSULATED_DATA", "BAD_DATA", "TIMESYNC", "SYSTEM_TIME"}
print("\n--- UNPARSED types (not consumed by io_layer) ---", flush=True)
for k in types:
    if k not in KNOWN:
        print(f"{k}: {samples.get(k)}", flush=True)
