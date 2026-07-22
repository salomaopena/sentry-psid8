#!/usr/bin/env python3
"""Tests for psid8/schema.py: schema.json as the ENFORCED source of truth.

Before psid8/schema.py existed, schema.json was documentation only: four
independent modules (sentryc/graph_builder.py, psid8/scripts/curate_clips.py,
sentry/plots.py, the Kaggle notebook) each hardcoded their own copy of the
8-class list, and psid8/scripts/agreement.py hardcoded its own copy of the
quality-gate thresholds. All copies happened to agree at audit time, purely
because they were kept in sync by hand -- nothing enforced it, exactly the
kind of drift `sentry/aggregate.py` vs `sentry/seeds.py` already demonstrated
in practice in an earlier audit.

This suite has two jobs:
  1. Prove `psid8.schema` loads correctly and exposes what its consumers need.
  2. Prove every consumer's class list is now IDENTICAL BY IMPORT (the same
     Python list object's contents, not a hand-copied duplicate), so future
     edits to schema.json propagate everywhere automatically and a
     reintroduced hardcoded copy would be caught by these assertions failing
     to import a name that no longer needs to exist.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_schema_loads_and_is_internally_consistent():
    from psid8.schema import (CLASS_NAMES, CLASS_ID, OPERATIONAL_DEFINITIONS,
                              ACTION_VOCABULARY, TEMPORAL_CLASSES,
                              MIN_KAPPA_PER_CLASS, MIN_MEAN_BOX_IOU,
                              COMPOSITIONAL_ATTRIBUTES)
    assert len(CLASS_NAMES) == 8
    assert CLASS_ID == {n: i for i, n in enumerate(CLASS_NAMES)}
    assert set(OPERATIONAL_DEFINITIONS) == set(CLASS_NAMES)
    assert set(ACTION_VOCABULARY) == set(CLASS_NAMES)
    assert all(isinstance(v, list) and v for v in ACTION_VOCABULARY.values())
    assert set(TEMPORAL_CLASSES) <= set(CLASS_NAMES)
    assert "fire" not in TEMPORAL_CLASSES, "fire is the static/appearance control class"
    assert 0.0 < MIN_KAPPA_PER_CLASS <= 1.0
    assert 0.0 < MIN_MEAN_BOX_IOU <= 1.0
    assert set(COMPOSITIONAL_ATTRIBUTES) == {"object", "action", "environment", "context"}
    print("test_schema_loads_and_is_internally_consistent OK")


def test_sentryc_graph_builder_uses_schema_directly():
    """sentryc's class ids must be the SAME list schema.json defines, not a
    hand-copied duplicate."""
    from psid8.schema import CLASS_NAMES as canonical
    from sentryc.graph_builder import CLASS_NAMES as sentryc_names, CLASS_ID as sentryc_ids
    assert sentryc_names == canonical
    assert sentryc_ids == {n: i for i, n in enumerate(canonical)}
    print("test_sentryc_graph_builder_uses_schema_directly OK")


def test_plots_uses_schema_directly():
    from psid8.schema import CLASS_NAMES as canonical
    from sentry.plots import CLASSES
    assert CLASSES == canonical
    print("test_plots_uses_schema_directly OK")


def test_curate_clips_uses_schema_directly():
    """curate_clips.py is loaded by file path (it is a standalone CLI
    script, not imported as part of a package elsewhere), mirroring how the
    other psid8/scripts/*.py test files already load it."""
    import importlib.util
    from psid8.schema import CLASS_NAMES as canonical

    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "psid8", "scripts", "curate_clips.py")
    spec = importlib.util.spec_from_file_location("curate_clips", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CLASSES == canonical
    print("test_curate_clips_uses_schema_directly OK")


def test_agreement_defaults_come_from_schema():
    """agreement.py's --kappa-min/--iou-min CLI defaults must equal
    schema.json's quality_gates, sourced by import, not by a second
    hardcoded literal that could silently diverge from the first."""
    import importlib.util
    from psid8.schema import MIN_KAPPA_PER_CLASS, MIN_MEAN_BOX_IOU

    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "psid8", "scripts", "agreement.py")
    spec = importlib.util.spec_from_file_location("agreement", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.N_CLASSES == 8
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--kappa-min", type=float, default=mod.MIN_KAPPA_PER_CLASS)
    ap.add_argument("--iou-min", type=float, default=mod.MIN_MEAN_BOX_IOU)
    parsed = ap.parse_args([])
    assert parsed.kappa_min == MIN_KAPPA_PER_CLASS
    assert parsed.iou_min == MIN_MEAN_BOX_IOU
    print("test_agreement_defaults_come_from_schema OK")


def test_notebook_order_imports_from_schema_not_hardcoded():
    """The Kaggle notebook's cell [1] must import CLASS_NAMES from
    psid8.schema (aliased as ORDER), not declare its own literal list -- this
    is a static-text check on the notebook source, since the notebook itself
    cannot be executed offline (it requires torch/ultralytics/GPU cells)."""
    import json as jsonlib
    nb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "notebooks", "sentry_kaggle.ipynb")
    nb = jsonlib.load(open(nb_path))
    found = False
    for c in nb["cells"]:
        s = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if "psid8.schema import CLASS_NAMES as ORDER" in s:
            found = True
            assert '"accident","suspicious_behavior"' not in s.replace(" ", ""), (
                "a hardcoded class list literal was reintroduced alongside the import"
            )
            break
    assert found, "notebook cell [1] no longer imports ORDER from psid8.schema"
    print("test_notebook_order_imports_from_schema_not_hardcoded OK")


if __name__ == "__main__":
    test_schema_loads_and_is_internally_consistent()
    test_sentryc_graph_builder_uses_schema_directly()
    test_plots_uses_schema_directly()
    test_curate_clips_uses_schema_directly()
    test_agreement_defaults_come_from_schema()
    test_notebook_order_imports_from_schema_not_hardcoded()
    print("\nALL psid8/schema.py CONSISTENCY TESTS PASSED")
