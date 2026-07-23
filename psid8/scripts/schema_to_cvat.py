#!/usr/bin/env python3
"""Generate a CVAT-importable label specification from psid8/schema.json.

psid8/schema.json is our own format (classes + compositional attribute
vocabularies + quality gates), not something CVAT understands directly --
there is no "upload schema.json to CVAT" step. CVAT expects its own label
specification format (a JSON array of label objects, each with its shape type
and its attribute definitions), which you paste into the project's label
constructor in "Raw" mode.

This script performs that translation FROM psid8.schema (the single source of
truth; see psid8/schema.py), so the 8 classes and their attribute vocabularies
never have to be hand-typed a second time into the CVAT UI -- exactly the kind
of duplication that caused real drift elsewhere in this project before
psid8/schema.py existed (see ARCHITECTURE.md section 6, item 7).

Usage:
    python psid8/scripts/schema_to_cvat.py --out cvat_labels.json

Then in CVAT: create a new Project -> Constructor -> "Raw" tab -> paste the
contents of cvat_labels.json -> Done. Every one of the 8 classes becomes a
CVAT label (rectangle shape), each with `object`, `action` (class-specific
vocabulary), `environment`, `context` and `event_id` attributes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from psid8.schema import load_schema

# A fixed, readable color per class (CVAT accepts any hex color; these are
# just distinct enough to tell classes apart at a glance in the CVAT UI).
_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
          "#911eb4", "#46f0f0", "#f032e6"]


def build_cvat_labels(schema: dict) -> list[dict]:
    """Translate schema.json into CVAT's raw label-constructor JSON."""
    comp = schema["compositional_attributes"]
    object_values = comp["object"]["values"]
    environment_values = comp["environment"]["values"]
    context_values = comp["context"]["values"]

    labels = []
    for i, cls in enumerate(schema["classes"]):
        labels.append({
            "name": cls["name"],
            "color": _COLORS[i % len(_COLORS)],
            "type": "rectangle",
            "attributes": [
                {
                    "name": "object",
                    "input_type": "select",
                    "mutable": False,
                    "default_value": object_values[0],
                    "values": object_values,
                },
                {
                    "name": "action",
                    "input_type": "select",
                    "mutable": False,
                    # class-specific vocabulary, per schema.json's
                    # classes[*].action_vocabulary -- NOT the global list
                    "default_value": cls["action_vocabulary"][0],
                    "values": cls["action_vocabulary"],
                },
                {
                    "name": "environment",
                    "input_type": "select",
                    "mutable": False,
                    "default_value": environment_values[0],
                    "values": environment_values,
                },
                {
                    "name": "context",
                    "input_type": "select",
                    "mutable": False,
                    # "abnormal" is the correct default for a real, annotated
                    # incident; hard negatives (VIDEO_ANNOTATION_GUIDE.md
                    # section 7) carry no event annotation at all, so this
                    # attribute value never needs to be "normal" in practice
                    # -- it exists mainly so the schema stays complete/self-
                    # describing rather than because annotators pick it often.
                    "default_value": "abnormal",
                    "values": context_values,
                },
                {
                    # Matches exactly what coco_to_yolo.py reads:
                    # a["attributes"]["event_id"] (see sentry.stageb_train /
                    # psid8/scripts/coco_to_yolo.py::_yolo_lines_and_attrs).
                    # Annotators type a short id (e.g. "ev1") shared by every
                    # box belonging to the same event within a clip, so
                    # multiple boxes link into one temporal track.
                    "name": "event_id",
                    "input_type": "text",
                    "mutable": False,
                    "default_value": "ev1",
                },
            ],
        })
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="cvat_labels.json")
    args = ap.parse_args()

    schema = load_schema()
    labels = build_cvat_labels(schema)
    json.dump(labels, open(args.out, "w"), indent=1, ensure_ascii=False)

    print(f"wrote {args.out}: {len(labels)} CVAT labels "
         f"(one per PSID-8 class, from schema.json v{schema.get('version')})")
    for lb in labels:
        attr_names = [a["name"] for a in lb["attributes"]]
        print(f"  {lb['name']:22s} attributes: {attr_names}")
    print(f"\nIn CVAT: New Project -> Constructor -> \"Raw\" -> paste {args.out}")


if __name__ == "__main__":
    main()
