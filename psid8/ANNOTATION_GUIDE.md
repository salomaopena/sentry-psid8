# Annotation Guide - PSID-8 (v0.2)

Annotation unit: **bounding box per frame**, linked to an `event_id` that
persists from the first to the last frame of the event. Multi-label is allowed
(a box may carry more than one class, e.g., accident + fall).

## Principles

1. **Annotate the event, not the visual category.** A person lying down is NOT a
   fall; annotate fall only if the transition was observed in the clip.
2. **Annotate visible evidence.** For fire, the box covers visible flame/smoke.
3. **Context comes from scenario metadata** (restricted zones, hours) - fill the
   `context` attribute accordingly; never invent undocumented context.
4. **Mandatory compositional attributes** per event: object, action,
   environment, context (vocabularies in `schema.json`, imported from the CVAT
   config `attributes.json`).
5. **Hard negatives are part of the job**: register them in the manifest with a
   `hard_negative` tag and the reason (e.g., "person lying, no transition").

## Tooling flow

CVAT (label config = attributes.json) -> export **COCO 1.0** (never YOLO:
attributes would be lost) -> `coco_to_yolo.py` -> `build_splits.py` ->
`agreement.py` on the double-annotated sample (>=20%) -> `dataset_stats.py` ->
`integrity_check.py`. Disagreements: third-annotator adjudication, logged in
`adjudications.csv`.

## Per-class rules (summary; full operational definitions in schema.json)

- **accident**: first contact -> scene at rest; include all parties.
- **suspicious_behavior**: requires a pattern (>=2 passes, prolonged
  observation, access testing, following). A single pass is NOT suspicious.
- **crime**: start of hostile action -> parties separate; a visible weapon is
  annotated as part of the carrier's box.
- **fire**: box on flame/smoke; re-annotate as the region grows.
- **intrusion**: any posture; starts at the perimeter crossing.
- **suspicious_object**: starts when the carrier leaves the frame/vicinity;
  the scenario's T_abandon threshold lives in metadata (default 30 s).
- **fall**: loss of balance -> rest on the ground; standing up ends the event.
- **vandalism**: damage in progress; finished graffiti with no author in scene
  is NOT an event.
