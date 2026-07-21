#!/usr/bin/env python3
"""Tests for sentry/train.py.

These require torch, opencv-python and ultralytics (all already declared
dependencies). They are skipped with a clear message if any is unavailable,
rather than failing the whole suite in an environment that only needs the
torch-free parts of the repository.

What is verified, and why each matters:

  * test_extract_feat_list_handles_both_ultralytics_shapes - pure-tensor unit
    test, no model needed. Ultralytics changed the training-mode output shape
    between versions actually seen in this project (plain feature list vs a
    {"boxes","scores","feats"} dict); this is the shim that keeps the training
    loop working across both.
  * test_stage_b_actually_updates_tfm_weights - THE regression test for the
    critical bug this file replaces: the previous version of sentry/train.py
    hard-coded `decoded = None`, so its loss branch never fired, `loss_total`
    stayed a Python float, and backward()/step() never ran for any epoch,
    while the script printed "epoch N: ok" and saved a checkpoint containing
    only the TFM's random initialization. This test trains a tiny real model
    for a couple of epochs and asserts that the TFM's parameter tensors
    actually changed.
  * test_zero_signal_epoch_raises_loudly - an epoch whose every window has no
    labeled frame must raise, not silently "succeed": this is the other half
    of the same correctness property (fail loud instead of fail silent).
  * test_batched_loop_uses_fewer_model_calls - measures, on tiny real data,
    that batching N clips together reduces the number of model() forward
    calls by a factor of N per timestep (T calls instead of N*T), which is
    the mechanism behind the wall-clock/memory improvement.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import cv2
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    _DEPS_OK = True
except ImportError as _e:
    _DEPS_OK = False
    _MISSING = str(_e)


def _skip(name):
    print(f"{name} SKIPPED (missing dependency: {_MISSING})")


def _build_synthetic_clips(root, clip_ids, n_frames=3, imgsz=64, label_frame=2):
    """Two-or-more clips, `n_frames` frames each, one labeled frame (class 6,
    a plausible box) per clip; the rest are background."""
    for cid in clip_ids:
        frames_dir = os.path.join(root, cid, "frames")
        labels_dir = os.path.join(root, cid, "labels")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        for i in range(1, n_frames + 1):
            img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), img)
            if i == label_frame:
                open(os.path.join(labels_dir, f"{i:05d}.txt"), "w").write(
                    "6 0.5 0.5 0.3 0.3\n")


def _fresh_sentry_model(hidden_ch=8, imgsz=64):
    from sentry.ultralytics_adapter import SentryYOLO
    base = YOLO("yolov8n.pt")   # small, already cached after the first test run
    model = SentryYOLO(base, hidden_ch=hidden_ch)
    model.reset_stream()
    model(torch.zeros(1, 3, imgsz, imgsz))
    model.train()
    model.freeze_base()
    for p in model.tfm.parameters():
        p.requires_grad_(True)
    model.reset_stream()
    return model


def test_extract_feat_list_handles_both_ultralytics_shapes():
    if not _DEPS_OK:
        _skip("test_extract_feat_list_handles_both_ultralytics_shapes")
        return
    from sentry.train import extract_feat_list

    a, b, c = torch.zeros(1), torch.zeros(2), torch.zeros(3)
    assert extract_feat_list([a, b, c]) == [a, b, c]
    assert extract_feat_list((a, b, c)) == [a, b, c]
    assert extract_feat_list({"boxes": None, "scores": None, "feats": [a, b, c]}) == [a, b, c]
    try:
        extract_feat_list({"boxes": None})
    except KeyError as e:
        assert "feats" in str(e)
    else:
        raise AssertionError("a dict without 'feats' must raise KeyError, not silently misbehave")
    print("test_extract_feat_list_handles_both_ultralytics_shapes OK")


def test_stage_b_actually_updates_tfm_weights():
    if not _DEPS_OK:
        _skip("test_stage_b_actually_updates_tfm_weights")
        return
    from sentry.data import VideoClipDataset, collate_batched
    from sentry.train import build_criterion, run_stage_b, set_seed

    set_seed(0)
    with tempfile.TemporaryDirectory() as d:
        clips_root = os.path.join(d, "clips")
        clip_ids = ["clipA", "clipB"]
        _build_synthetic_clips(clips_root, clip_ids)

        ds = VideoClipDataset(clips_root, clip_ids, window=3, stride=3,
                              imgsz=64, event_only=True)
        assert len(ds) == 2
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batched)

        model = _fresh_sentry_model()
        criterion = build_criterion(model.base)
        before = {k: v.clone() for k, v in model.tfm.state_dict().items()}

        history = run_stage_b(model, criterion, dl, epochs=2, lr=1e-2,
                              lambda_tc=1.0, device=torch.device("cpu"),
                              use_amp=False, log_fn=lambda *a, **k: None)

        after = model.tfm.state_dict()
        n_changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
        assert n_changed == len(before), (
            f"only {n_changed}/{len(before)} TFM tensors changed; Stage B is "
            f"not actually training (the exact silent-failure mode this file "
            f"replaces)."
        )
        assert all(torch.isfinite(v).all() for v in after.values())
        assert len(history) == 2
        for h in history:
            assert np.isfinite(h["det"]) and np.isfinite(h["tc"])
        print("test_stage_b_actually_updates_tfm_weights OK "
              f"({n_changed}/{len(before)} tensors changed)")


def test_zero_signal_epoch_raises_loudly():
    if not _DEPS_OK:
        _skip("test_zero_signal_epoch_raises_loudly")
        return
    from sentry.data import VideoClipDataset, collate_batched
    from sentry.train import build_criterion, run_stage_b, set_seed

    set_seed(0)
    with tempfile.TemporaryDirectory() as d:
        clips_root = os.path.join(d, "clips")
        clip_ids = ["clipA", "clipB"]
        _build_synthetic_clips(clips_root, clip_ids)

        # window=1, event_only=False, then keep only unlabeled-frame windows:
        # every window in this dataset carries zero labeled frames.
        ds = VideoClipDataset(clips_root, clip_ids, window=1, stride=1,
                              imgsz=64, event_only=False)
        ds.samples = [s for s in ds.samples
                     if not any("00002.jpg" in fp for fp in s[1])]
        assert len(ds) > 0
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batched)

        model = _fresh_sentry_model()
        criterion = build_criterion(model.base)

        try:
            run_stage_b(model, criterion, dl, epochs=1, lr=1e-3, lambda_tc=1.0,
                       device=torch.device("cpu"), use_amp=False,
                       log_fn=lambda *a, **k: None)
        except RuntimeError as e:
            assert "zero labeled frames" in str(e)
        else:
            raise AssertionError(
                "an all-background epoch must raise, not silently report success "
                "(this is the failure mode the old sentry/train.py had)"
            )
        print("test_zero_signal_epoch_raises_loudly OK")


def test_batched_loop_uses_fewer_model_calls():
    if not _DEPS_OK:
        _skip("test_batched_loop_uses_fewer_model_calls")
        return
    from sentry.data import VideoClipDataset, collate_batched
    from sentry.train import build_criterion, run_stage_b, set_seed

    set_seed(0)
    with tempfile.TemporaryDirectory() as d:
        clips_root = os.path.join(d, "clips")
        clip_ids = [f"clip{i}" for i in range(4)]
        _build_synthetic_clips(clips_root, clip_ids, n_frames=4, label_frame=2)

        batch_size = 2
        ds = VideoClipDataset(clips_root, clip_ids, window=4, stride=4,
                              imgsz=64, event_only=True)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_batched)
        n_windows, T = len(ds), 4
        expected_calls = n_windows // batch_size * T   # T calls per mini-batch

        model = _fresh_sentry_model()
        criterion = build_criterion(model.base)

        calls = {"n": 0}
        orig_forward = model.forward
        def counting_forward(x):
            calls["n"] += 1
            return orig_forward(x)
        model.forward = counting_forward

        run_stage_b(model, criterion, dl, epochs=1, lr=1e-3, lambda_tc=1.0,
                   device=torch.device("cpu"), use_amp=False,
                   log_fn=lambda *a, **k: None)

        assert calls["n"] == expected_calls, (
            f"expected {expected_calls} model() calls (T={T} per mini-batch of "
            f"{batch_size} clips), got {calls['n']}: batching is not reducing "
            f"the call count as designed."
        )
        naive_calls = n_windows * T   # what the old per-clip loop would have done
        print(f"test_batched_loop_uses_fewer_model_calls OK "
              f"({calls['n']} calls vs {naive_calls} for the old per-clip loop, "
              f"a {naive_calls / calls['n']:.1f}x reduction)")


if __name__ == "__main__":
    test_extract_feat_list_handles_both_ultralytics_shapes()
    test_stage_b_actually_updates_tfm_weights()
    test_zero_signal_epoch_raises_loudly()
    test_batched_loop_uses_fewer_model_calls()
    print("\nALL sentry/train.py TESTS PASSED (or cleanly skipped)")
