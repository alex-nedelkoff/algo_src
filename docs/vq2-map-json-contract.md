# VQ2 course map JSON contract (v1)

Contract for the 6-gate map JSON (producer: Janahan; consumer: `vq2/map_ingest.py`).
Send corrections/extensions as a v2 proposal — the loader rejects unknown versions.

## Frame (non-negotiable, declared in the file)

- **Spawn frame**: origin at drone spawn, `x` = downcourse, `y` = right, `z` = down (NED-style).
- Units: **meters**.
- `pos` is the **aperture center** (the 1.5 m inner square the judge scores), NOT the
  outer structure center or banner center.
- `normal` is the unit vector the drone flies **along** when crossing correctly
  (i.e. pointing downcourse through the gate, roughly +x for gate 1).

## Schema

```json
{
  "version": 1,
  "frame": "spawn:x-downcourse,y-right,z-down",
  "units": "m",
  "source": "free-text provenance (runs, method, date)",
  "gates": [
    {
      "id": "G1",
      "route_order": 1,
      "pos": [6.3, 0.0, -1.35],
      "normal": [1.0, 0.0, 0.0],
      "aperture_m": 1.5,
      "confidence": "ticked | observed | inferred",
      "pos_sigma_m": 0.2
    }
  ]
}
```

## Rules the loader enforces

1. `version == 1`, `frame` exactly as above, `units == "m"`.
2. `gates` non-empty; `route_order` values are exactly `1..N` with no gaps or dups.
3. `id` unique, non-empty strings.
4. `pos` finite 3-vector; `normal` finite 3-vector with norm within 1 ± 0.05
   (loader re-normalizes).
5. `aperture_m` in (0.5, 5.0); defaults to 1.5 if omitted.
6. `confidence` one of `ticked|observed|inferred`; `pos_sigma_m` finite >= 0
   (optional, default 0.5 for observed, 1.0 for inferred, 0.1 for ticked).

## Known anchor cross-checks (loader warns, does not reject)

- G1 aperture ~[6.2-6.4, 0.0, -1.35] (judge-verified pad lock).
- G2 ~[11.7, 5.2, -1.35] (judge-verified, fg62).

If the delivered map disagrees with a judge-verified anchor by > 1.0 m the loader
flags it loudly — resolve before flight, the judge evidence wins.
