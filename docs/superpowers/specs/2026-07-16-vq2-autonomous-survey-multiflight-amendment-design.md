# VQ2 Autonomous Survey Multi-Flight Amendment

## Status

This is a normative amendment to
`2026-07-16-vq2-autonomous-survey-map-localization-design.md` at commit
`4e9e079e`. Where the base specification describes one map-building survey A
and one validation survey B, this amendment takes precedence.

The autonomous-control architecture is unchanged: the survey uses GateNet,
FastGate, cyan-ribbon guidance, and short-horizon inertial propagation with
DPVO disabled. It observes and bypasses gates rather than threading them.

## Collection sets

"A" is a map-building set of three independent autonomous flights:

- **A1:** left-side gate bypass near racing height.
- **A2:** right-side bypass for complementary parallax.
- **A3:** a targeted gap-filling pass selected from the A1/A2 coverage report.
  It may adjust height or observation-fan width but remains inside GateNet's
  race-relevant viewing envelope.

**B1** is an independent race-like validation flight. B1 is sealed before map
thresholds are frozen and cannot contribute images, landmarks, associations,
feature choices, optimizer settings, or acceptance thresholds to map A.

If B1 is later folded into a production map, that produces a new map artifact
and invalidates B1 as holdout evidence. A new independent sealed validation
flight must pass before the expanded map can be approved.

Partial or aborted A flights may contribute only the segments whose manifests
and quality reports pass corpus validation. A partial or aborted B1 flight
cannot approve uncovered course segments.

## Exploratory-flight control

The first survey has no trusted metric world map. Control therefore remains
local and image-relative:

1. The existing deduplicated IMU/ESKF and bounded attitude controller
   stabilize the vehicle.
2. CPU cyan-ribbon extraction supplies the forward course tangent.
3. GateNet supplies multi-candidate gate geometry, metric gate-relative range,
   bearing, orientation, and outer-frame extent.
4. A gate is eligible only when temporal tracking and ribbon geometry agree.
5. At stand-off, the vehicle brakes and collects a bounded lateral observation
   fan.
6. The bypass controller creates a temporary gate-local frame and flies on the
   configured pass side. Its initial centerline offset is the observed outer
   half-width plus 1.0 m and never less than 2.5 m from gate center. Replay must
   prove this envelope before flight; changing it creates a new survey profile.
7. FastGate supplies low-latency aperture/structure continuity but does not
   select persistent gate identity or route order.
8. After the gate is behind, the vehicle brakes, performs a bounded forward
   scan, and reacquires the ribbon before resuming.

The existing compressed course map may restrict the yaw-search cone. It may
not command metric distance or override local perception. Judge ticks are not
used for survey progression; an unexpected tick means the bypass envelope
failed and causes a controlled abort.

Lost ribbon, ambiguous candidates, stale vision, disagreement between
detectors, or failed post-bypass reacquisition produces hover and landing.
There is no general blind-forward mode.

## Promotion sequence

The live survey ladder in the base specification is replaced after its
two-gate step by:

1. full left-bypass pass A1;
2. full right-bypass pass A2;
3. frozen A1/A2 coverage report;
4. coverage-directed pass A3;
5. map-A construction and threshold freeze;
6. independent race-like validation pass B1.

Only one survey-profile variable changes per flight. A3's route is declared
from the frozen A1/A2 report before launch. B1 is not opened by mapping tools
until map A and its acceptance policy are immutable and hashed.

## Disk policy

The existing per-pass limits remain: reserve 20 GB on C:, cap each pass at
1 GB and five minutes, retain the source JPEG stream, and avoid duplicate
frame trees. At the measured 75-87 MB/minute rate, four five-minute passes
consume roughly 1.5-2 GB before derived products, well inside the current disk
budget.

Every pass has a unique directory and manifest. Derived GateNet, DPVO, feature,
and map artifacts reference source frames by hash. Regenerable intermediates
may be removed only after the corresponding immutable artifact and validation
report are verified.

## Implementation-plan consequence

The survey-acquisition plan must implement pass profiles, pass-set membership,
coverage reporting, and sealed-holdout enforcement. The map-builder plan must
accept A1/A2/A3 as training inputs and reject B1 anywhere outside validation.
The validation plan must reject any map whose input or tuning digests include
B1, and must require a new holdout after B1 is incorporated into a later map.

