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

## Landmark extension (v1-pillar, 2026-07-18)

Sibling file `pillar_map_*.json` for station-pillar landmarks (producer: the
tick-anchored/pad-window survey pipeline; consumer: `vq2/map_ingest.py::
load_pillar_map`). Pillars carry a lit 2-digit station-number panel at a
shared height; numbers DUPLICATE across aisle twins, so `number` is not a
key — `id` is.

```json
{
  "version": "pillar-1",
  "frame": "janahan-v1 datum (offline GateNet range scale; NOT the flight pad-lock datum — reconcile before flight use)",
  "units": "m",
  "z_panel": -7.16,
  "landmarks": [
    {
      "id": "22b",
      "number": "22",
      "pos": [23.54, 4.29, -7.16],
      "confidence": "ticked | observed | inferred",
      "pos_sigma_m": 0.5,
      "source": "free-text provenance (survey method, corpora, date)"
    }
  ],
  "quarantined": [
    { "number": "06|22", "pos": [19.7, 4.0, -7.16], "reason": "why it is not a landmark yet" }
  ]
}
```

Loader rules: `version == "pillar-1"`; `landmarks` non-empty; `id` unique;
`number` = 1-2 digit string (duplicates allowed — twins); `pos` finite
3-vector; same `confidence`/`pos_sigma_m` semantics as gates (ticked =
tick-anchored survey, observed = pad-window truth-pose survey, inferred =
grid extrapolation). `quarantined` entries are carried for bookkeeping and
NEVER returned as landmarks.

### markings[] (v3 semantics, 2026-07-19)

One physical pillar carries its station number at MULTIPLE heights (lit
top panel, face panels, mid-height vertical text — Alex's correction).
From map v3 on, a landmark entry is a PHYSICAL pillar: `pos` is the pillar
axis xy with z = the top-panel height, and an optional `markings` list
records the per-height sub-features:

```json
"markings": [
  { "kind": "top_panel", "z": -7.16 },
  { "kind": "lower", "z": -5.5 },
  { "kind": "lower", "z": null }
]
```

`kind` ∈ {`top_panel`, `lower`}; `z` finite or `null` (known-to-exist but
unsurveyed). Same-number entries lying on a common view-ray family are ONE
pillar — collapse them (v2's 22b+22c). Consumers: only `top_panel`
markings justify elevation-range measurement models; any marking supports
azimuth (the bearing to a vertical pillar is height-independent).
