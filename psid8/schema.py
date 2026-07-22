"""Single loader for psid8/schema.json.

This module makes `schema.json` load-bearing: every consumer above now
imports `CLASS_NAMES`/`CLASS_ID` (or the quality gates) from here, so there is
exactly one place a class list or threshold can be edited, and
`tests/test_schema.py` asserts nothing has silently hardcoded its own copy
that could drift.
"""
from __future__ import annotations

import json
import os

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def load_schema(path: str = _SCHEMA_PATH) -> dict:
    """Load the raw schema.json document."""
    return json.load(open(path, encoding="utf-8"))


_schema = load_schema()

#: Canonical class names, in their canonical id order (index == class id).
CLASS_NAMES: list[str] = [c["name"] for c in _schema["classes"]]

#: name -> id, derived from the same source, never hand-maintained separately.
CLASS_ID: dict[str, int] = {c["name"]: c["id"] for c in _schema["classes"]}

#: id -> operational definition (one sentence; the terse, machine-readable
#: spec -- ANNOTATION_GUIDE.md / VIDEO_ANNOTATION_GUIDE.md expand on this in
#: prose for annotators, and are expected to phrase it more fully, not verbatim).
OPERATIONAL_DEFINITIONS: dict[str, str] = {
    c["name"]: c["operational_definition"] for c in _schema["classes"]
}

#: id -> class-specific action vocabulary (closed list).
ACTION_VOCABULARY: dict[str, list[str]] = {
    c["name"]: c["action_vocabulary"] for c in _schema["classes"]
}

#: Which classes are temporally defined (require a transition, not just
#: appearance) vs. appearance-defined. Used by the "primary fusion" reasoning
#: in sentryc/graph_builder.py's docstring context, and by the paper's
#: seed-eligible/consequence-only split -- kept here as the single source
#: rather than re-derived independently.
TEMPORAL_CLASSES: list[str] = [c["name"] for c in _schema["classes"] if c.get("temporal")]

#: Quality gates for inter-annotator agreement (Phase 0 calibration and the
#: ongoing double-annotation sample). psid8/scripts/agreement.py reads these
#: as its CLI defaults instead of hardcoding them a second time.
QUALITY_GATES: dict = _schema["quality_gates"]
MIN_KAPPA_PER_CLASS: float = QUALITY_GATES["min_kappa_per_class"]
MIN_MEAN_BOX_IOU: float = QUALITY_GATES["min_mean_box_iou"]

#: Compositional attribute vocabularies (object / action / environment / context).
COMPOSITIONAL_ATTRIBUTES: dict = _schema["compositional_attributes"]

assert len(CLASS_NAMES) == 8, f"expected 8 canonical classes, schema.json has {len(CLASS_NAMES)}"
assert CLASS_ID == {n: i for i, n in enumerate(CLASS_NAMES)}, (
    "schema.json's class ids must match their list position; the rest of the "
    "codebase assumes id == index"
)
