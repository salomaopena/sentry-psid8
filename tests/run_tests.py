"""Offline tests for SENTRY/PSID-8 (pure numpy). Run: python tests/run_tests.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentry.tubes import TubeLinker, confirm_events, iou_xyxy
from sentry.metrics import tiou, match_events, event_auc, event_prf, bootstrap_ci, streams_per_gpu
from psid8.scripts.build_splits import build_splits, verify


def test_tube_linking_and_alerts():
    lk = TubeLinker(iou_thr=0.5, max_gap=2)
    # persistent object (class 3=fire) for 6 frames + 1-frame noise (class 2)
    for t in range(6):
        dets = [{"bbox": [10, 10, 50, 50], "class_id": 3, "score": 0.9,
                 "evidence": {"motion_gate_mean": 0.4}}]
        if t == 2:
            dets.append({"bbox": [200, 200, 220, 220], "class_id": 2, "score": 0.6})
        lk.update(t, dets)
    tubes = lk.finalize()
    assert len(tubes) == 2, f"expected 2 tubes, got {len(tubes)}"
    alerts = confirm_events(tubes, {3: 2, 2: 5})
    assert len(alerts) == 1 and alerts[0]["class_id"] == 3, "n_c persistence failed"
    a = alerts[0]
    assert a["t_start"] == 0 and a["t_end"] == 5
    assert abs(a["evidence_terms"]["motion_gate_mean"] - 0.4) < 1e-9
    assert len(a["confidence_trajectory"]) == 6
    print("test_tube_linking_and_alerts OK")


def test_tube_gap_break():
    lk = TubeLinker(iou_thr=0.5, max_gap=1)
    lk.update(0, [{"bbox": [0, 0, 10, 10], "class_id": 6, "score": 0.8}])
    lk.update(4, [{"bbox": [0, 0, 10, 10], "class_id": 6, "score": 0.8}])  # gap 4 > 1
    tubes = lk.finalize()
    assert len(tubes) == 2, "gap should break the tube"
    print("test_tube_gap_break OK")


def test_event_metrics():
    gts = [{"class_id": 6, "t_start": 10, "t_end": 30},
           {"class_id": 3, "t_start": 0, "t_end": 5}]
    preds = [{"class_id": 6, "t_start": 12, "t_end": 28, "confidence": 0.9},   # TP
             {"class_id": 6, "t_start": 100, "t_end": 120, "confidence": 0.3}] # FP
    assert abs(tiou(10, 30, 12, 28) - (17 / 21)) < 1e-9
    pairs, fp, fn = match_events(preds, gts)
    assert len(pairs) == 1 and len(fp) == 1 and len(fn) == 1
    auc = event_auc(preds, gts)
    assert 0.0 <= auc <= 1.0 and auc == auc
    prf = event_prf(preds, gts)
    assert prf[6]["P"] == 0.5 and prf[6]["R"] == 1.0
    assert prf[3]["R"] == 0.0                       # missed fire event
    m, lo, hi = bootstrap_ci([0.8, 0.9, 0.85, 0.7], n_boot=200)
    assert lo <= m <= hi
    assert streams_per_gpu(96.0) == 3
    print("test_event_metrics OK")


def test_splits_camera_disjoint_and_coverage():
    rng = np.random.default_rng(0)
    clips = []
    cid = 0
    for scen in ["rua", "estacionamento", "entrada"]:
        for cam in range(8):
            cam_id = f"{scen}_cam{cam}"
            for k in range(3):
                classes = sorted(set(rng.choice(8, size=rng.integers(0, 3)).tolist()))
                clips.append({"clip_id": f"c{cid}", "camera_id": cam_id,
                              "scenario": scen, "classes": classes,
                              "n_frames": int(rng.integers(100, 500))})
                cid += 1
    # guarantee coverage of every class across varied cameras
    for c in range(8):
        for j, scen in enumerate(["rua", "estacionamento", "entrada"]):
            clips.append({"clip_id": f"c{cid}", "camera_id": f"{scen}_cam{c % 8}",
                          "scenario": scen, "classes": [c], "n_frames": 200})
            cid += 1
    splits, assign, _ = build_splits(clips, seed=0)
    verify(clips, splits)
    total = sum(len(v) for v in splits.values())
    assert total == len(clips)
    print("test_splits_camera_disjoint_and_coverage OK",
          {s: len(v) for s, v in splits.items()})


def test_iou_helpers():
    assert abs(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    print("test_iou_helpers OK")


if __name__ == "__main__":
    test_iou_helpers()
    test_tube_linking_and_alerts()
    test_tube_gap_break()
    test_event_metrics()
    test_splits_camera_disjoint_and_coverage()
    print("\nALL OFFLINE TESTS PASSED")
