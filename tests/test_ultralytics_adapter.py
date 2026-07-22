#!/usr/bin/env python3
"""Tests for sentry/ultralytics_adapter.py: decode_eval_output and
run_clip_to_alerts across YOLO generations.

Requires torch + ultralytics (declared dependencies); skipped cleanly if
unavailable. Network access is needed the first time a model is downloaded
(cached afterwards).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from ultralytics import YOLO
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _MISSING = str(_e)


def _skip(name):
    print(f"{name} SKIPPED (missing dependency: {_MISSING})")


def test_decode_eval_output_requires_explicit_end2end():
    """`end2end` must be a required, explicit argument: in the Ultralytics
    version this project targets, BOTH end2end (YOLO26) and non-end2end
    (YOLOv8/11) models return the same `(tensor, dict)` tuple shape in eval
    mode, so neither the Python type nor a shape heuristic can reliably
    distinguish them. `decode_eval_output` delegates the actual dispatch to
    Ultralytics' own `non_max_suppression(..., end2end=...)`, the same
    official code path `DetectionPredictor.postprocess` uses, instead of
    re-deriving the distinction independently."""
    if not _DEPS_OK:
        _skip("test_decode_eval_output_requires_explicit_end2end")
        return
    from sentry.ultralytics_adapter import decode_eval_output
    try:
        decode_eval_output(torch.zeros(1, 5, 10))
    except TypeError:
        pass
    else:
        raise AssertionError("end2end must be a required argument, not silently defaulted")
    print("test_decode_eval_output_requires_explicit_end2end OK")


def test_decode_eval_output_scores_are_valid_probabilities():
    """Regression test for a real bug found while building this: an earlier
    version of decode_eval_output dispatched on tuple/dict type, which
    misclassified YOLOv8's (grid_tensor, aux_dict) eval output as the
    end-to-end (pre-decoded) case, silently treating raw grid logits as
    already-decoded [x,y,x,y,score,class] rows -- producing "scores" outside
    [0,1] (observed: ~30.5) without any error. This test would have caught
    that: every returned score must be a valid probability."""
    if not _DEPS_OK:
        _skip("test_decode_eval_output_scores_are_valid_probabilities")
        return
    from sentry.ultralytics_adapter import SentryYOLO, decode_eval_output

    for weights, expect_e2e in [("yolov8n.pt", False), ("yolo26n.pt", True)]:
        try:
            base = YOLO(weights)
        except Exception as e:
            print(f"  {weights} SKIPPED ({e})")
            continue
        model = SentryYOLO(base, hidden_ch=8)
        model.reset_stream(); model(torch.zeros(1, 3, 320, 320))
        model.eval(); model.reset_stream()
        is_e2e = getattr(model.base, "end2end", False)
        assert is_e2e == expect_e2e, (
            f"{weights}: expected end2end={expect_e2e}, model reports {is_e2e} "
            f"(Ultralytics may have changed this attribute; re-verify before trusting the pipeline)"
        )
        with torch.no_grad():
            raw = model(torch.zeros(1, 3, 320, 320))
        dets = decode_eval_output(raw, end2end=is_e2e, conf=0.0)
        assert isinstance(dets, list)
        for d in dets:
            assert set(d) == {"bbox", "score", "class_id"}
            assert len(d["bbox"]) == 4
            assert 0.0 <= d["score"] <= 1.0, (
                f"{weights}: score {d['score']} is not a valid probability; "
                f"the end2end dispatch is misclassifying this output again."
            )
        print(f"  {weights}: OK ({len(dets)} dets, all scores in [0,1])")
    print("test_decode_eval_output_scores_are_valid_probabilities OK")


def test_run_clip_to_alerts_smoke_both_generations():
    """End-to-end smoke test of the real call path (what the notebook's
    inference cells use): must not crash, must derive `end2end` automatically
    from `sentry.base.end2end`, and must return a list, for both a YOLOv8 and
    a YOLO26 backbone."""
    if not _DEPS_OK:
        _skip("test_run_clip_to_alerts_smoke_both_generations")
        return
    from sentry.ultralytics_adapter import SentryYOLO, run_clip_to_alerts

    for weights in ["yolov8n.pt", "yolo26n.pt"]:
        try:
            base = YOLO(weights)
        except Exception as e:
            print(f"  {weights} SKIPPED ({e})")
            continue
        model = SentryYOLO(base, hidden_ch=8)
        model.reset_stream(); model(torch.zeros(1, 3, 320, 320))
        frames = [torch.zeros(1, 3, 320, 320) for _ in range(3)]
        alerts = run_clip_to_alerts(model, frames, conf_per_class={c: 0.0 for c in range(8)},
                                    min_persistence={c: 1 for c in range(8)})
        assert isinstance(alerts, list)
        print(f"  {weights}: OK ({len(alerts)} alerts, structural)")
    print("test_run_clip_to_alerts_smoke_both_generations OK")


if __name__ == "__main__":
    test_decode_eval_output_requires_explicit_end2end()
    test_decode_eval_output_scores_are_valid_probabilities()
    test_run_clip_to_alerts_smoke_both_generations()
    print("\nALL sentry/ultralytics_adapter.py TESTS PASSED (or cleanly skipped)")
